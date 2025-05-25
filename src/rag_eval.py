import re
import string
from collections import Counter
import tiktoken

try:
    QWEN_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    QWEN_TOKENIZER = tiktoken.get_encoding("qwen3")
except Exception:
    print("Cảnh báo: Không tìm thấy encoding 'qwen2' cho tiktoken. Sử dụng 'cl100k_base' làm fallback.")
    print("Hãy đảm bảo bạn đã cài đặt phiên bản tiktoken mới nhất và 'qwen2' được hỗ trợ, hoặc chỉ định encoding chính xác.")
    QWEN_TOKENIZER = tiktoken.get_encoding("cl100k_base")

def normalize_answer(s):
    """Chuẩn hóa chuỗi: viết thường, xóa khoảng trắng thừa.
    Bỏ xóa dấu câu và mạo từ tiếng Anh để tokenizer LLM xử lý."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text): 
        return text.lower()
    
    return white_space_fix(lower(s))

def tokenize(s):
    normalized_s = normalize_answer(s)
    if not normalized_s:
        return []
    token_ids = QWEN_TOKENIZER.encode(normalized_s)
    tokens = [QWEN_TOKENIZER.decode_single_token_bytes(token_id).decode('utf-8', 'ignore') for token_id in token_ids]
    return tokens

def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def recall_score(prediction, ground_truth):
    """Recall so với ground truth"""
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    return num_same / len(gt_tokens)

def evaluate(predictions, references):
    """predictions, references: danh sách cùng chiều"""
    total = len(predictions)
    em, f1, recall = 0, 0, 0

    for p, r in zip(predictions, references):
        em += exact_match_score(p, r)
        f1 += f1_score(p, r)
        recall += recall_score(p, r)

    return {
        "Exact Match": round(100 * em / total, 2),
        "F1 Score": round(100 * f1 / total, 2),
        "Answer Recall": round(100 * recall / total, 2)
    }
