import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss 
import pickle 

CHUNKED_DATA_DIR = "../chunked_data_semantic_custom"         
INDEX_DIR = "vector_store_AITeamVN"  

MODEL_NAME = 'AITeamVN/Vietnamese_Embedding' 

EMBEDDING_DIM = 1024 

os.makedirs(INDEX_DIR, exist_ok=True)

# Đường dẫn file lưu trữ
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.faiss")
CHUNK_METADATA_PATH = os.path.join(INDEX_DIR, "chunk_metadata.pkl") 

def load_all_chunks():
    """Tải tất cả nội dung từ các file chunk."""
    all_chunks_content = [] 
    chunk_metadata_list = [] 
    
    print(f"Đang tải các chunks từ thư mục: {CHUNKED_DATA_DIR}")
    filenames = [f for f in os.listdir(CHUNKED_DATA_DIR) if f.endswith(".txt")]
    if not filenames:
        print(f"Không tìm thấy file chunk nào trong {CHUNKED_DATA_DIR}. Hãy đảm bảo bạn đã chạy bước chunking.")
        return [], []

    for filename in sorted(filenames): 
        filepath = os.path.join(CHUNKED_DATA_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines and lines[0].startswith("Nguồn:"):
                    content = "".join(lines[2:]) 
                else:
                    content = "".join(lines)
                
                if content.strip():
                    all_chunks_content.append(content.strip())
                    chunk_metadata_list.append({"source_file": filename, "text": content.strip()})
        except Exception as e:
            print(f"Lỗi khi đọc file chunk {filepath}: {e}")
            
    print(f"Đã tải {len(all_chunks_content)} chunks.")
    return all_chunks_content, chunk_metadata_list

def build_vector_store():
    """Xây dựng và lưu trữ vector store sử dụng FAISS."""
    chunks_text, chunks_metadata = load_all_chunks()
    
    if not chunks_text:
        print("Không có chunk nào để tạo index. Kết thúc.")
        return

    print(f"Đang tải model embedding: {MODEL_NAME}...")
    try:

        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device="cuda")  
    except Exception as e:
        print(f"Lỗi khi tải model SentenceTransformer: {e}")
        print("Hãy đảm bảo bạn đã cài đặt sentence-transformers và Pytorch (nếu dùng GPU).")
        return

    print("Đang tạo embeddings cho các chunks...")

    chunk_embeddings = model.encode(chunks_text, batch_size=4, convert_to_tensor=False, show_progress_bar=True, truncation=True, max_length=2000) 
    
    print(f"Đã tạo {chunk_embeddings.shape[0]} embeddings với kích thước {chunk_embeddings.shape[1]}.")

    if chunk_embeddings.shape[1] != EMBEDDING_DIM:
        print(f"LỖI: Kích thước embedding ({chunk_embeddings.shape[1]}) không khớp với EMBEDDING_DIM ({EMBEDDING_DIM}) đã khai báo.")
        print(f"Vui lòng kiểm tra lại model '{MODEL_NAME}' hoặc cập nhật EMBEDDING_DIM.")
        return

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    
    print("Đang thêm embeddings vào FAISS index...")
    index.add(chunk_embeddings.astype(np.float32)) 
    
    print(f"FAISS index đã được xây dựng với {index.ntotal} vectors.")
    
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"Đã lưu FAISS index vào: {FAISS_INDEX_PATH}")
    
    with open(CHUNK_METADATA_PATH, 'wb') as f_meta:
        pickle.dump(chunks_metadata, f_meta)
    print(f"Đã lưu metadata của chunks vào: {CHUNK_METADATA_PATH}")

if __name__ == "__main__":
    print("Bắt đầu xây dựng Vector Store...")
    build_vector_store()
    print("Hoàn thành xây dựng Vector Store.")