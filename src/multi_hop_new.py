import os
import requests
import logging
import time 
from retriever import FaissRetriever, MODEL_NAME as RETRIEVER_MODEL_NAME 
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(threadName)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

INDEX_DIR = "vector_store_AITeamVN" 
FAISS_INDEX_PATH = f"{INDEX_DIR}/faiss_index.idx"
CHUNK_METADATA_PATH = f"{INDEX_DIR}/chunk_metadata.pkl"
load_dotenv()

class OpenRouterGenerator:
    def __init__(self, api_key=None, model_name="qwen/qwen3-30b-a3b:free", system_message=None, role_for_answer="assistant"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Cần cung cấp OpenRouter API key qua biến môi trường OPENROUTER_API_KEY hoặc tham số api_key.")
        self.model_name = model_name
        self.default_system_message = system_message or (
            "Bạn là một trợ lý AI hữu ích. "
            "Chỉ trả lời bằng tiếng Việt. "
            "Chỉ sử dụng thông tin có trong ngữ cảnh được cung cấp để trả lời câu hỏi. "
            "Không được nói các cụm như 'Theo thông tin trong NGỮ CẢNH', 'Theo tài liệu', 'Theo thông tin được cung cấp' hoặc các câu tương tự. "
            "Chỉ trả lời trực tiếp vào nội dung câu hỏi. "
            "Nếu không tìm thấy thông tin trong ngữ cảnh, hãy trả lời: 'Tôi không tìm thấy thông tin này trong tài liệu được cung cấp.' "
            "Tuyệt đối không suy diễn, không tự bổ sung hoặc dự đoán thông tin ngoài ngữ cảnh."
        )
        self.role_for_answer = role_for_answer 

    def _build_messages_for_generation(self, question_text, retrieved_contexts, custom_system_message=None):
        context_str = "\n\n---\n\n".join([ctx['text'] for ctx in retrieved_contexts if ctx.get('text')]) if retrieved_contexts else "Không có thông tin ngữ cảnh nào được truy xuất."
        
        user_instruction = (
            "Chỉ sử dụng thông tin trong NGỮ CẢNH dưới đây để trả lời câu hỏi. "
            "Không được nói các cụm như 'Theo thông tin trong NGỮ CẢNH', 'Theo tài liệu', 'Theo thông tin được cung cấp' hoặc các câu tương tự. "
            "Chỉ trả lời trực tiếp vào nội dung câu hỏi, không nhắc lại nguồn hoặc ngữ cảnh. "
            "Không được suy diễn, không tự bổ sung thông tin ngoài ngữ cảnh. "
            "Nếu không có thông tin, hãy trả lời đúng như hướng dẫn hệ thống.\n\n"
            f"Câu hỏi: {question_text}\n\nNGỮ CẢNH:\n{context_str}"
        )
        return [
            {"role": "system", "content": custom_system_message or self.default_system_message},
            {"role": "user", "content": user_instruction}
        ]

    def _build_messages_for_planner(self, original_question, history_context_summary, previous_hop_answer=None):
        planner_system_message = (
            "Bạn là một trợ lý nghiên cứu thông minh. Nhiệm vụ của bạn là phân tích câu hỏi gốc và những thông tin đã tìm được để quyết định xem có cần tìm thêm thông tin không. "
            "Nếu cần tìm thêm, hãy đặt một câu hỏi cụ thể, rõ ràng để truy vấn thêm. Câu hỏi này phải giúp làm sáng tỏ hoặc bổ sung những khía cạnh còn thiếu của câu hỏi gốc. "
            "Nếu bạn thấy thông tin đã đủ để trả lời câu hỏi gốc một cách toàn diện, hãy trả lời chính xác bằng cụm từ: [ĐỦ THÔNG TIN]. "
            "Câu hỏi theo dõi phải bằng tiếng Việt."
        )
        
        user_content_for_planner = f"""Câu hỏi gốc cần trả lời: "{original_question}"

Thông tin đã tìm được cho đến hiện tại (tóm tắt một phần):
---
{history_context_summary if history_context_summary else "Chưa có thông tin nào được tìm thấy."}
---
"""
        if previous_hop_answer:
            user_content_for_planner += f"""\nCâu trả lời dựa trên thông tin hiện tại (từ hop trước): "{previous_hop_answer}"
---
"""
        user_content_for_planner += "\nPhân tích và câu hỏi tiếp theo (hoặc trả lời [ĐỦ THÔNG TIN]):"
        
        return [
            {"role": "system", "content": planner_system_message},
            {"role": "user", "content": user_content_for_planner}
        ]

    def _call_openrouter_api(self, messages, max_tokens=1024, temperature=0.1, top_p=0.9, **kwargs):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost", 
            "X-Title": "RAG MultiHop Test"      
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs 
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120) 
            response.raise_for_status() 
            data = response.json()
            logging.info(f"Data: {data}")

            if not data.get("choices") or not isinstance(data["choices"], list) or len(data["choices"]) == 0:
                logging.error(f"API response không hợp lệ (thiếu 'choices'): {data}")
                return "Lỗi API: Phản hồi không hợp lệ."

            message_obj = data["choices"][0].get("message", {})
            answer = (message_obj.get("content") or "").strip()

            if not answer and message_obj.get("reasoning"): 
                logging.warning(f"API response không có 'content' trong message. Sử dụng 'reasoning'.")
                answer = (message_obj.get("reasoning") or "").strip()
            
            if not answer and message_obj.get("refusal"): 
                logging.warning(f"API response không có 'content' và 'reasoning'. Sử dụng 'refusal'.")
                answer = (message_obj.get("refusal") or "").strip()

            if not answer:
                logging.error(f"Không có content, reasoning hoặc refusal hợp lệ trong message. Data: {data}")
                return "Không nhận được câu trả lời từ API."
            return answer
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"Lỗi HTTP khi gọi OpenRouter API: {http_err.response.status_code} - {http_err.response.text}", exc_info=False) # Không cần full traceback cho lỗi HTTP
            return f"Lỗi API: {http_err.response.status_code}."
        except Exception as e:
            logging.error(f"Lỗi khác khi gọi OpenRouter API: {e}", exc_info=True)
            return "Đã xảy ra lỗi khi hệ thống tạo câu trả lời (OpenRouter)."


    def generate_final_answer(self, query_text, retrieved_contexts, max_tokens=1024, temperature=0.1, top_p=0.9):
        """Sinh câu trả lời cuối cùng dựa trên context đã thu thập."""
        messages = self._build_messages_for_generation(query_text, retrieved_contexts)
        answer = self._call_openrouter_api(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        
        no_info_indicators = ["không tìm thấy thông tin này trong tài liệu được cung cấp", "không có thông tin", "không thể trả lời"] # Cụm từ chính xác
        if any(ind.lower() == answer.lower().strip().rstrip('.') for ind in no_info_indicators):
             answer = "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
        return answer

    def generate_follow_up_query(self, original_question, history_context_summary, previous_hop_answer=None, max_tokens=150, temperature=0.1):
        """Sinh câu hỏi theo dõi hoặc quyết định dừng."""
        messages = self._build_messages_for_planner(original_question, history_context_summary, previous_hop_answer)
        raw_planner_output = self._call_openrouter_api(messages, max_tokens=max_tokens, temperature=temperature, top_p=0.9)
        
        logging.info(f"Planner LLM Output (thô): {raw_planner_output}")
        
        if "[ĐỦ THÔNG TIN]" in raw_planner_output.upper() or \
           "enough information" in raw_planner_output.lower() or \
           raw_planner_output.strip() == "" or \
           len(raw_planner_output.strip()) < 10 :
            logging.info("Planner quyết định đã đủ thông tin hoặc không thể tạo câu hỏi hợp lệ.")
            return None

        prefixes_to_remove = [
            "câu hỏi tiếp theo là:", "câu hỏi tiếp theo:", "để làm rõ hơn, hãy hỏi:",
            "phân tích và câu hỏi tiếp theo:", "phân tích và câu hỏi:",
            "phân tích:", "câu hỏi:",
        ]
        cleaned_query = raw_planner_output.strip()
        for prefix in prefixes_to_remove:
            if cleaned_query.lower().startswith(prefix):
                cleaned_query = cleaned_query[len(prefix):].strip()
        
        if "?" not in cleaned_query:
            if "." in cleaned_query or "!" in cleaned_query:
                 logging.warning(f"Planner output có vẻ không phải câu hỏi: '{cleaned_query}'. Coi như không có câu hỏi theo dõi.")
                 return None
        
        if len(cleaned_query) > 5: 
            logging.info(f"Câu hỏi theo dõi được tạo: '{cleaned_query}'")
            return cleaned_query
        
        logging.warning(f"Không thể trích xuất câu hỏi theo dõi hợp lệ từ output của planner: '{raw_planner_output}'")
        return None


def multi_hop_rag_improved(original_query, retriever, generator, max_hops=1, initial_k=4, follow_up_k=3, delay_between_api_calls=1):
    logging.info(f"\n=== Bắt đầu Multi-Hop RAG cho câu hỏi: \"{original_query}\" ===")
    
    all_unique_contexts = {} 
    current_query_for_retrieval = original_query
    answer_from_previous_hop = None 

    for hop in range(1, max_hops + 1):
        logging.info(f"\n--- Hop {hop}/{max_hops} ---")
        logging.info(f"Query cho retrieval ở hop này: \"{current_query_for_retrieval}\"")

        k_for_this_hop = initial_k if hop == 1 else follow_up_k
        try:
            retrieved_contexts_this_hop = retriever.retrieve(current_query_for_retrieval, k=k_for_this_hop)
            if retrieved_contexts_this_hop:
                new_contexts_added_count = 0
                for ctx in retrieved_contexts_this_hop:
                    if ctx['text'] not in all_unique_contexts:
                        all_unique_contexts[ctx['text']] = ctx 
                        new_contexts_added_count += 1
                logging.info(f"Hop {hop}: Truy xuất được {len(retrieved_contexts_this_hop)} chunks, thêm {new_contexts_added_count} chunks mới vào tổng context.")
            else:
                logging.warning(f"Hop {hop}: Không truy xuất được context mới.")
        except Exception as e:
            logging.error(f"Lỗi retrieve ở Hop {hop}: {e}", exc_info=True)

        current_context_list_for_llm = list(all_unique_contexts.values())

        if hop < max_hops and current_context_list_for_llm: 
            logging.info(f"Hop {hop}: Tạo câu trả lời thử nghiệm cho planner...")
            answer_from_previous_hop = generator.generate_final_answer(
                original_query,
                current_context_list_for_llm,
                temperature=0.1 
            )
            logging.info(f"Hop {hop}: Câu trả lời thử nghiệm: \"{answer_from_previous_hop}\"")
            logging.info(f"Dừng {delay_between_api_calls} giây trước khi gọi API cho planner...")
            time.sleep(delay_between_api_calls)

        if hop < max_hops:
            history_summary_for_planner = "\n".join([ctx['text'][:1000] + "..." for ctx in current_context_list_for_llm]) # Tóm tắt ngắn
            
            next_query_candidate = generator.generate_follow_up_query(
                original_question=original_query,
                history_context_summary=history_summary_for_planner,
                previous_hop_answer=answer_from_previous_hop 
            )
            
            if next_query_candidate:
                current_query_for_retrieval = next_query_candidate
            else:
                logging.info(f"Hop {hop}: Planner quyết định dừng. Chuyển sang sinh câu trả lời cuối cùng.")
                break 
        else:
            logging.info(f"Đã đạt số hop tối đa ({max_hops}).")

    logging.info("\n--- Sinh câu trả lời cuối cùng (sau tất cả các hop) ---")
    if not all_unique_contexts:
        logging.warning("Không có context nào được thu thập sau tất cả các hop.")
    
    final_answer = generator.generate_final_answer(original_query, list(all_unique_contexts.values()))
    logging.info(f"Câu trả lời cuối cùng (Multi-Hop): \"{final_answer}\"")
    return final_answer


if __name__ == "__main__":
    logging.info("\n--- Thử nghiệm hệ thống Multi-hop RAG (Retriever + OpenRouter Generator) - CẢI TIẾN ---")

    API_KEYS = os.getenv("OPENROUTER_API_KEY") 
    if not API_KEYS:
        logging.error("Biến môi trường OPENROUTER_API_KEY_QWEN chưa được đặt.")
        exit()
        
    MODEL_OPEN_ROUTER = "qwen/qwen3-30b-a3b:free" 

    try:
        retriever = FaissRetriever(RETRIEVER_MODEL_NAME, FAISS_INDEX_PATH, CHUNK_METADATA_PATH)
        generator_instance = OpenRouterGenerator(
            api_key=API_KEYS,
            model_name=MODEL_OPEN_ROUTER
        )

    
        sample_queries = [
            "Viện Trí tuệ nhân tạo thuộc Trường Đại học Công nghệ được thành lập vào ngày tháng năm nào?",
            'Phòng ban nào của Trường Đại học Công nghệ (UET) chịu trách nhiệm tiếp nhận và xử lý các thông tin điều chỉnh từ sinh viên?',
            "Trường Đại học Công nghệ (ĐHQGHN) đã đạt được thành tích gì tại Olympic Toán học Sinh viên và Học sinh năm 2023?"

        ]

        for sq in sample_queries:
            final_answer = multi_hop_rag_improved(sq, retriever, generator_instance, max_hops=10, initial_k=4, follow_up_k=3, delay_between_api_calls=2.5) # Tăng delay lên 2s
            print(f"\nQUESTION: {sq}\nFINAL ANSWER: {final_answer}\n" + "="*50)


    except Exception as e:
        logging.error(f"Lỗi trong quá trình thử nghiệm hệ thống Multi-hop RAG: {e}", exc_info=True)