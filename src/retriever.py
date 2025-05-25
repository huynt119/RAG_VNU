import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

INDEX_DIR = "vector_store_AITeamVN2"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.idx")
CHUNK_METADATA_PATH = os.path.join(INDEX_DIR, "chunk_metadata.pkl")
MODEL_NAME = 'AITeamVN/Vietnamese_Embedding' 

class FaissRetriever:
    def __init__(self, model_name, index_path, metadata_path):
        print(f"Retriever: Đang tải model embedding: {model_name}...")
        try:
            self.embedding_model = SentenceTransformer(model_name, device='cuda', trust_remote_code=True)
        except Exception as e:
            print(f"Lỗi khi tải model SentenceTransformer cho Retriever: {e}")
            raise
            
        print(f"Retriever: Đang tải FAISS index từ: {index_path}...")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Không tìm thấy file FAISS index: {index_path}. Hãy chạy script build_vector_store.py trước.")
        self.index = faiss.read_index(index_path)
        
        print(f"Retriever: Đang tải metadata của chunks từ: {metadata_path}...")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Không tìm thấy file metadata: {metadata_path}. Hãy chạy script build_vector_store.py trước.")
        with open(metadata_path, 'rb') as f:
            self.chunk_metadata = pickle.load(f) 
            
        print(f"Retriever đã sẵn sàng với {self.index.ntotal} chunks trong index.")

    def retrieve(self, query_text, k=3):
        """
        Truy xuất k chunks liên quan nhất đến câu hỏi.
        Trả về list các dict, mỗi dict chứa 'text' và 'source_file'.
        """
        print(f"\nRetriever: Đang tạo embedding cho câu hỏi: '{query_text}'")
        query_embedding = self.embedding_model.encode([query_text], convert_to_tensor=False)
        print("retriever metric type: ", self.index.metric_type)
        print(f"Retriever: Đang tìm kiếm {k} chunks gần nhất...")
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        retrieved_chunks = []
        print("Retriever: Các chunks được truy xuất:")
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            if idx < 0 or idx >= len(self.chunk_metadata): 
                print(f"  - Lỗi: Index {idx} không hợp lệ. Bỏ qua.")
                continue

            chunk_info = self.chunk_metadata[idx]
            retrieved_chunks.append({
                "text": chunk_info["text"],
                "source_file": chunk_info["source_file"],
                "score": 1 - dist if self.index.metric_type == faiss.METRIC_L2 else dist 
            })
            print(f"  - Chunk {i+1} (Index: {idx}, Nguồn: {chunk_info['source_file']}, Distance/Score: {dist:.4f}):")
            print(f"    \"\"\"{chunk_info['text'][:3000]}...\"\"\"")
            
        return retrieved_chunks

if __name__ == "__main__":
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNK_METADATA_PATH)):
        print("Chưa có Vector Store. Vui lòng chạy script build_vector_store.py trước khi chạy retriever.")
    else:
        retriever = FaissRetriever(MODEL_NAME, FAISS_INDEX_PATH, CHUNK_METADATA_PATH)
        
        sample_query_2 = "Lễ khai giảng năm học 2019-2020 diễn ra vào ngày nào?"
        print("\n--- Thử nghiệm Retriever ---")
        
        retrieved_docs_2 = retriever.retrieve(sample_query_2, k=5)
        for doc in retrieved_docs_2:
            print(f"Nguồn: {doc['source_file']}, Score: {doc.get('score', 'N/A')}")
            print(doc['text'])
            print("-" * 20)