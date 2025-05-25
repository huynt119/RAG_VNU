import os
import csv
import random
import logging
from multi_hop_new import multi_hop_rag_improved, OpenRouterGenerator
from rag_eval import exact_match_score, f1_score, recall_score
from dotenv import load_dotenv

from retriever import FaissRetriever, MODEL_NAME as RETRIEVER_MODEL_NAME

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(threadName)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

INDEX_DIR = "vector_store_AITeamVN2"
FAISS_INDEX_PATH = f"{INDEX_DIR}/faiss_index.idx"
CHUNK_METADATA_PATH = f"{INDEX_DIR}/chunk_metadata.pkl"

load_dotenv()
API_KEYS = os.getenv("OPENROUTER_API_KEY") 
MODEL_OPEN_ROUTER = "qwen/qwen3-30b-a3b:free"

NUM_RANDOM_QUESTIONS = 2     

def read_questions(file_path):
    questions = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "question": row["question"],
                "reference_answer": row["reference_answer"]
            })
    return questions

def read_questions_from_txt(question_path, answer_path):
    questions = []
    with open(question_path, encoding="utf-8") as fq, open(answer_path, encoding="utf-8") as fa:
        question_lines = [line.strip() for line in fq if line.strip()]
        answer_lines = [line.strip() for line in fa if line.strip()]
        min_len = min(len(question_lines), len(answer_lines))
        for i in range(min_len):
            questions.append({
                "question": question_lines[i],
                "reference_answer": answer_lines[i]
            })
    return questions

def calc_average_metrics(results):
    total_em = sum(res["EM"] for res in results)
    total_f1 = sum(res["F1"] for res in results)
    total_recall = sum(res["Recall"] for res in results)
    n = len(results)
    avg_em = total_em / n if n else 0
    avg_f1 = total_f1 / n if n else 0
    avg_recall = total_recall / n if n else 0
    return avg_em, avg_f1, avg_recall

if __name__ == "__main__":
    retriever = FaissRetriever(RETRIEVER_MODEL_NAME, FAISS_INDEX_PATH, CHUNK_METADATA_PATH)
    generator = OpenRouterGenerator(api_key=API_KEYS, model_name=MODEL_OPEN_ROUTER)

    QA_FOLDER = ""
    QUESTION_TXT = os.path.join(QA_FOLDER, "selected_questions.txt")
    ANSWER_TXT = os.path.join(QA_FOLDER, "selected_answers.txt")
    questions = read_questions_from_txt(QUESTION_TXT, ANSWER_TXT)

    results = []
    for idx, item in enumerate(questions, 1):
        question = item["question"]
        ref_answer = item["reference_answer"]
        retrieved_contexts = retriever.retrieve(question, k=3)
        gen_answer = multi_hop_rag_improved(
            question, retriever, generator,
            max_hops=10, initial_k=4, follow_up_k=3, delay_between_api_calls=2.5
        )

        em = exact_match_score(gen_answer, ref_answer)
        f1 = f1_score(gen_answer, ref_answer)
        recall = recall_score(gen_answer, ref_answer)
        print(f"EM: {em}, F1: {f1:.2f}, Recall: {recall:.2f}")

        results.append({
            "idx": idx,
            "question": question,
            "reference_answer": ref_answer,
            "generated_answer": gen_answer,
            "EM": em,
            "F1": f1,
            "Recall": recall
        })

    with open("result.txt", "w", encoding="utf-8") as f:
        for res in results:
            f.write(f"\n--- Câu hỏi {res['idx']} ---\n")
            f.write(f"Câu hỏi: {res['question']}\n")
            f.write(f"Đáp án tham chiếu: {res['reference_answer']}\n")
            f.write(f"Đáp án mô hình:    {res['generated_answer']}\n")
            f.write(f"EM: {res['EM']}, F1: {res['F1']:.2f}, Recall: {res['Recall']:.2f}\n")
        avg_em, avg_f1, avg_recall = calc_average_metrics(results)
        f.write("\n=== Trung bình trên tất cả câu hỏi ===\n")
        f.write(f"EM trung bình: {avg_em:.2f}\n")
        f.write(f"F1 trung bình: {avg_f1:.2f}\n")
        f.write(f"Recall trung bình: {avg_recall:.2f}\n")