import os
import PyPDF2
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss_2"
documents_path = "vectorstores/documents2.json" 

# Hàm để trích xuất văn bản từ file PDF
def extract_text_from_pdfs(pdf_folder):
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_folder, filename)
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                documents.append(text.strip(),)
                print(f"Extracted text from: {filename}")
    return documents

# Hàm tạo database từ file PDF
def create_db_from_files():
    # Nếu chưa có cơ sở dữ liệu, ta sẽ tạo mới
    global documents
    documents = extract_text_from_pdfs(pdf_data_path)
    
    with open(documents_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)
    
    embeddings = model.encode(documents, batch_size=32, show_progress_bar=True).astype('float32')
    embedding_size = embeddings.shape[1]
    print("Embeddings generated.")
    
    # Tạo FAISS index
    index = faiss.IndexFlatL2(embedding_size)
    index.add(embeddings)
    
    # Lưu index vào file
    faiss.write_index(index, vector_db_path)
    print("FAISS index created and saved.")

# Gọi hàm để tạo DB 
create_db_from_files()
