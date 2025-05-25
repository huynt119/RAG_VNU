import os
import re

CRAWLED_DATA_DIR = "tuyen_sinh_vnu/tuyen_sinh_vnu"  
CLEANED_DATA_DIR = "cleaned_data"       

if not os.path.exists(CLEANED_DATA_DIR):
    os.makedirs(CLEANED_DATA_DIR)

def clean_text(text):
    """Thực hiện các thao tác làm sạch văn bản cơ bản."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    
    cleaned_text = "\n".join(lines)
    
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    cleaned_text = re.sub(r'\[(?:edit|sửa|citation needed|cần dẫn nguồn)\]', '', cleaned_text, flags=re.IGNORECASE)
    
    cleaned_text = "\n".join([line for line in cleaned_text.splitlines() if not re.match(r'^[=\-_*#\s]+$', line)])

    return cleaned_text.strip()

def process_all_crawled_files():
    for filename in os.listdir(CRAWLED_DATA_DIR):
        if filename.endswith(".txt"):
            input_filepath = os.path.join(CRAWLED_DATA_DIR, filename)
            output_filepath = os.path.join(CLEANED_DATA_DIR, filename) 
            
            try:
                with open(input_filepath, 'r', encoding='utf-8') as f_in:
                    raw_text = f_in.read()
                
                cleaned_content = clean_text(raw_text)
                
                if cleaned_content: 
                    with open(output_filepath, 'w', encoding='utf-8') as f_out:
                        f_out.write(cleaned_content)
                    print(f"Đã làm sạch và lưu: {output_filepath}")
                else:
                    print(f"Bỏ qua file rỗng sau khi làm sạch: {input_filepath}")
            except Exception as e:
                print(f"Lỗi khi xử lý file {input_filepath}: {e}")

if __name__ == "__main__":
    print("Bắt đầu làm sạch dữ liệu...")
    process_all_crawled_files()
    print("Hoàn thành làm sạch dữ liệu.")