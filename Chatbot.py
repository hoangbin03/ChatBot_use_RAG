from flask import Flask, jsonify, session
from flask import Blueprint, render_template, request, redirect, url_for, flash
import json
import faiss
from sentence_transformers import SentenceTransformer
from gradio_client import Client
import speech_recognition as sr
import numpy as np
import os
from gtts import gTTS
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_db_connection
from decorators import login_required
from datetime import datetime
import re

chatbot_bp = Blueprint('chatbot', __name__)

# Khởi tạo các thành phần toàn cục
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load tài liệu
with open("vectorstores/documents.json", 'r', encoding='utf-8') as f:
    documents = json.load(f)

# Load FAISS index
vector_db_path = "vectorstores/db_faiss"
index = faiss.read_index(vector_db_path)

# Kết nối Gradio Client (thay đổi API nếu cần)
client = Client("yuntian-deng/ChatGPT")

def build_prompt(chat_context, user_input, document_info):
    """Tạo prompt tối ưu để gửi đến mô hình ChatGPT."""
    initial_context = (
        "Bạn là một trợ lý thông minh và có thể trả lời các câu hỏi phức tạp."
        "Hãy trả lời người dùng một cách chính xác."
    )

    history = "\n".join(chat_context)

    prompt = (
        f"{initial_context}\n\n"
        f"Lịch sử hội thoại:\n"
        f"{history}\n\n"
        f"Thông tin tài liệu liên quan:\n"
        f"{document_info}\n\n"
        f"Câu hỏi của người dùng: {user_input}\n\n"
        "Câu trả lời của bạn:"
    )
    return prompt

def search_and_predict(user_input, chat_context):
    """Tìm kiếm tài liệu bằng FAISS + ranking cosine similarity, rồi gửi truy vấn đến LLM."""
    chat_context.append(f"User: {user_input}")

    # Tạo embedding cho truy vấn
    input_embedding = sentence_model.encode([user_input]).astype('float32')

    # Tìm top-k tài liệu từ FAISS
    k = 10
    D, I = index.search(input_embedding, k)

    # Lấy tài liệu liên quan (lọc những index hợp lệ)
    retrieved_docs = [(i, documents[i]) for i in I[0] if 0 <= i < len(documents)]

    if not retrieved_docs:
        combined_info = "Không tìm thấy tài liệu nào."
    else:
        # Tính điểm tương đồng cosine giữa truy vấn và tài liệu
        retrieved_embeddings = np.array([sentence_model.encode([doc]).astype('float32')[0] for _, doc in retrieved_docs])
        similarity_scores = cosine_similarity(input_embedding, retrieved_embeddings)[0]

        # Sắp xếp tài liệu theo độ tương đồng giảm dần
        sorted_docs = sorted(zip(similarity_scores, retrieved_docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, (_, doc) in sorted_docs[:5]]  

        combined_info = "\n".join(top_docs)

    # Xây dựng prompt
    prompt = build_prompt(chat_context, user_input, combined_info)

    # Gọi API ChatGPT
    result = client.predict(
        inputs=prompt,
        top_p=1,
        temperature=1,
        api_name="/predict"
    )

    # Lấy câu trả lời từ mô hình
    answer = result[0][-1][-1] if len(result) > 0 and len(result[0]) > 0 and len(result[0][-1]) > 0 else "Không tìm thấy câu trả lời."
    chat_context.append(f"ChatBot: {answer}")
    return answer

def create_new_conversation(user_id, title="Hội thoại mới"):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("INSERT INTO conversations (user_id, title, created_at) VALUES (%s, %s, %s)",
                   (user_id, title, datetime.now()))
    conn.commit()
    conversation_id = cursor.lastrowid
    conn.close()
    return conversation_id

def save_message(conversation_id, sender, message):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (conversation_id, sender, message, timestamp) VALUES (%s, %s, %s, %s)",
                   (conversation_id, sender, message, datetime.now()))
    conn.commit()
    conn.close()


def get_conversations(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Lấy danh sách các cuộc hội thoại của người dùng
    cursor.execute("SELECT id, title, created_at FROM conversations WHERE user_id = %s", (user_id,))
    conversations = cursor.fetchall()

    result = []
    for conversation in conversations:
        conversation_id = conversation[0]

        # Lấy tin nhắn của cuộc hội thoại
        cursor.execute("SELECT sender, message, timestamp FROM messages WHERE conversation_id = %s", (conversation_id,))
        messages = cursor.fetchall()

        # Thêm tin nhắn vào lịch sử cuộc hội thoại
        chat_history = [{"sender": msg[0], "message": msg[1], "timestamp": msg[2]} for msg in messages]
        
        # Đảm bảo trả về thông tin cuộc hội thoại cùng với chat_history
        result.append({
            "id": conversation_id,
            "title": conversation[1],
            "created_at": conversation[2],
            "messages": chat_history
        })

    conn.close()
    return result

def clean_text(text):
    # Loại bỏ tất cả ký tự không phải chữ cái, số, khoảng trắng và một số dấu cơ bản
    return re.sub(r'[^\w\s,.!?-]', '', text)

@chatbot_bp.route('/get_conversation/<int:conversation_id>')
def get_conversation(conversation_id):
    if 'user_id' not in session:
        flash("Bạn cần đăng nhập để xem cuộc hội thoại", "warning")
        return redirect(url_for('login.login'))
    
    user_id = session['user_id']
    
    # Kiểm tra xem hội thoại có thuộc về user_id này không
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM conversations WHERE id = %s", (conversation_id,))
    conversation_info = cursor.fetchone()
    
    if not conversation_info or conversation_info[0] != user_id:
        conn.close()
        flash("Bạn không có quyền truy cập vào hội thoại này", "danger")
        return redirect(url_for('chatbot.home'))
    
    # Nếu kiểm tra quyền thành công, tiến hành truy vấn tin nhắn
    cursor.execute("SELECT sender, message, timestamp FROM messages WHERE conversation_id = %s", (conversation_id,))
    messages = cursor.fetchall()
    conn.close()

    chat_history = [{"sender": msg[0], "message": msg[1], "timestamp": msg[2]} for msg in messages]
    chat_context = [f"{msg['sender']}: {msg['message']}" for msg in chat_history]
    return jsonify({"messages": chat_history,"chat_context": chat_context})

@chatbot_bp.route('/delete_conversation/<int:conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
    conn.commit()
    conn.close()
    return '', 204 


@chatbot_bp.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    user_input = data.get('user_input', '')
    chat_context = data.get('chat_context', [])
    conversation_id = data.get('conversation_id')

    # Nếu không có conversation_id thì tạo mới
    if not conversation_id:
        if 'user_id' not in session:
            return jsonify({'error': 'Người dùng chưa đăng nhập'}), 403
        user_id = session['user_id']
        conversation_id = create_new_conversation(user_id)

    # Gửi tin nhắn người dùng vào lịch sử
    save_message(conversation_id, 'User', user_input)

    # phản hồi từ chatbot
    response_text = search_and_predict(user_input, chat_context)
    cleaned_response =  clean_text(response_text)
    # Thêm phản hồi của bot vào cơ sở dữ liệu
    save_message(conversation_id, 'ChatBot', cleaned_response)

    return jsonify({
        'response': cleaned_response,
        'chat_context': chat_context,
        'conversation_id': conversation_id
    })

@chatbot_bp.route("/loadConversation")
def load_conversation():
    conversation_id = session.get("conversation_id")
    if not conversation_id:
        return jsonify([])

    messages = get_conversation(conversation_id)
    return jsonify(messages)

@chatbot_bp.route('/')
@login_required
def home():
    if 'user_id' in session:
        user_id = session.get('user_id',[])
        user_name = session.get('user_name', 'Người dùng')
        conversations = get_conversations(user_id)
        return render_template('Main.html',  user_name=user_name, conversations=conversations)

    
    flash("Bạn cần đăng nhập!", "warning")
    return redirect(url_for('login.login'))
