import os
import PyPDF2
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from flask import render_template, request, jsonify, Blueprint
import re

embedding = Blueprint('embedding', __name__)

# Khai báo model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Đường dẫn lưu trữ
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"
documents_path = "vectorstores/documents.json" 

# Đảm bảo thư mục tồn tại
os.makedirs(pdf_data_path, exist_ok=True)
os.makedirs("vectorstores", exist_ok=True)

# Danh sách stopwords tiếng Việt
vietnamese_stopwords = set([
    "và", "là", "của", "có", "cho", "với", "một", "những", "đã", "trong",
    "khi", "thì", "lại", "này", "đó", "nên", "ra", "ở", "từ", "được", "do",
    "để", "vì", "như", "sau", "đang", "các", "bị", "đến", "nếu", "hay", "mà",
    "rằng", "hoặc", "đi", "ai", "gì", "vẫn", "tại", "trên", "dưới", "nào"
])

def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text)  
    words = text.split()
    filtered_words = [word for word in words if word not in vietnamese_stopwords]
    return ' '.join(filtered_words)

# Trích xuất văn bản từ file PDF
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Load FAISS index
def load_existing_faiss():
    if os.path.exists(vector_db_path):
        return faiss.read_index(vector_db_path)
    return None

# Load văn bản đã lưu
def load_existing_documents():
    if os.path.exists(documents_path):
        with open(documents_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Xử lý upload file
@embedding.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"message": "Không có file nào được chọn!"}), 400
    
    files = request.files.getlist('files')
    existing_docs = load_existing_documents()
    
    new_texts = []
    for file in files:
        if file.filename.endswith(".pdf"):
            file_path = os.path.join(pdf_data_path, file.filename)
            file.save(file_path)

            raw_text = extract_text_from_pdf(file_path)
            processed_text = preprocess_text(raw_text)

            # Tránh trùng lặp nếu đã có processed_text
            if processed_text and all(processed_text != doc for doc in existing_docs):
                new_texts.append(processed_text)
                existing_docs.append(processed_text)
    
    if new_texts:
        embeddings = model.encode(new_texts,batch_size=32, show_progress_bar=True).astype('float32')
        index = load_existing_faiss()

        if index is None:
            index = faiss.IndexFlatL2(embeddings.shape[1])

        index.add(embeddings)
        faiss.write_index(index, vector_db_path)

        with open(documents_path, "w", encoding="utf-8") as f:
            json.dump(existing_docs, f, ensure_ascii=False, indent=4)

        return jsonify({"message": "Upload và cập nhật dữ liệu thành công!"})
    
    return jsonify({"message": "Không có dữ liệu mới để cập nhật!"})

@embedding.route('/embedding')
def embedding_page():
    return render_template('embedding.html')
