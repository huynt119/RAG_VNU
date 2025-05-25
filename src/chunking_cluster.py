import os
import nltk 
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken 
import torch
import traceback

CLEANED_DATA_DIR = "cleaned_data"
CHUNKED_SEMANTIC_DIR = "chunked_data_semantic_custom"

EMBEDDING_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
SIMILARITY_THRESHOLD = 0.5 


MAX_CHUNK_TOKENS = 2000 
MIN_CHUNK_TOKENS = 500  

TOKEN_ENCODING_NAME = "cl100k_base" 

if not os.path.exists(CHUNKED_SEMANTIC_DIR):
    os.makedirs(CHUNKED_SEMANTIC_DIR)

try:
    print(f"Đang tải embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer("dangvantuan/vietnamese-document-embedding", trust_remote_code=True,device='cuda' )

    print("Embedding model đã tải xong.")
    tokenizer_for_chunk_size = tiktoken.get_encoding(TOKEN_ENCODING_NAME)
except Exception as e:
    print(f"Lỗi khi tải model hoặc tokenizer: {e}")
    exit()

def num_tokens(text: str) -> int:
    """Đếm số token trong một chuỗi."""
    return len(tokenizer_for_chunk_size.encode(text))

def get_sentence_embeddings(sentences):
    """Tạo embedding cho một list các câu."""
    if not sentences:
        return np.array([])
    return embedding_model.encode(sentences, convert_to_tensor=False, show_progress_bar=False)

def semantic_chunk_text(text_content: str, source_filename: str):
    """
    Thực hiện semantic chunking cho một nội dung văn bản.
    """
    raw_sentences = nltk.sent_tokenize(text_content, language='english')
    
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10] 

    if not sentences:
        print(f"  Không có câu hợp lệ nào trong file: {source_filename}")
        return []

    sentence_embeddings = get_sentence_embeddings(sentences)
    if sentence_embeddings.ndim == 1: 
        if num_tokens(sentences[0]) > MIN_CHUNK_TOKENS :
             return [sentences[0]] if num_tokens(sentences[0]) <= MAX_CHUNK_TOKENS else []
        return []


    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        sim = cosine_similarity(sentence_embeddings[i].reshape(1, -1), 
                                sentence_embeddings[i+1].reshape(1, -1))[0,0]
        similarities.append(sim)

    split_points = [0] 
    for i, sim in enumerate(similarities):
        if sim < SIMILARITY_THRESHOLD:
            split_points.append(i + 1) 
    
    if split_points[-1] != len(sentences): 
        split_points.append(len(sentences))
    
    split_points = sorted(list(set(split_points)))

    final_chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0
    
    start_index = 0
    for i in range(len(sentences)):
        sentence_to_add = sentences[i]
        tokens_in_sentence = num_tokens(sentence_to_add)
        
        should_split_semantically = (i in split_points and i > start_index and current_chunk_sentences)
        
        if current_chunk_sentences and \
           (should_split_semantically or (current_chunk_tokens + tokens_in_sentence > MAX_CHUNK_TOKENS)):
            chunk_text = " ".join(current_chunk_sentences)
            if num_tokens(chunk_text) >= MIN_CHUNK_TOKENS:
                final_chunks.append(chunk_text)
            current_chunk_sentences = [sentence_to_add]
            current_chunk_tokens = tokens_in_sentence
            start_index = i 
        else:
            current_chunk_sentences.append(sentence_to_add)
            current_chunk_tokens += tokens_in_sentence
            
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        if num_tokens(chunk_text) >= MIN_CHUNK_TOKENS:
            final_chunks.append(chunk_text)
            
    return final_chunks


def semantic_chunk_all_cleaned_files():
    for filename in os.listdir(CLEANED_DATA_DIR):
        if filename.endswith(".txt"):
            base_name = os.path.splitext(filename)[0]
            existing_chunks = [f for f in os.listdir(CHUNKED_SEMANTIC_DIR) if f.startswith(base_name + "_semantic_custom_chunk_")]
            if existing_chunks:
                print(f"Bỏ qua file {filename}, đã xử lý rồi, tìm thấy {len(existing_chunks)} chunk.")
                continue
            input_filepath = os.path.join(CLEANED_DATA_DIR, filename)
            
            try:
                with open(input_filepath, 'r', encoding='utf-8') as f_in:
                    cleaned_content = f_in.read()
                
                if not cleaned_content.strip():
                    print(f"Bỏ qua file rỗng: {input_filepath}")
                    continue

                print(f"File: {filename} - Bắt đầu semantic chunking...")
                chunks = semantic_chunk_text(cleaned_content, filename)
                
                print(f"File: {filename}, Số chunks (semantic) tạo ra: {len(chunks)}")

                for i, chunk_text in enumerate(chunks):
                    if not chunk_text.strip():
                        continue

                    chunk_filename = f"{os.path.splitext(filename)[0]}_semantic_custom_chunk_{i+1}.txt"
                    output_filepath = os.path.join(CHUNKED_SEMANTIC_DIR, chunk_filename)
                    
                    with open(output_filepath, 'w', encoding='utf-8') as f_out:
                        f_out.write(f"Nguồn: {filename}\n\n")
                        f_out.write(chunk_text)

            except Exception as e:
                print(f"Lỗi khi semantic chunking file {input_filepath}: {e}")
                traceback.print_exc()
                continue

if __name__ == "__main__":
    import torch 

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Đang tải tài nguyên 'punkt' cho NLTK...")
        nltk.download('punkt')
        print("'punkt' đã tải xong.")

    print("\nBắt đầu phân đoạn tài liệu dựa trên ngữ nghĩa (custom)...")
    semantic_chunk_all_cleaned_files()
    print("Hoàn thành phân đoạn tài liệu dựa trên ngữ nghĩa (custom).")