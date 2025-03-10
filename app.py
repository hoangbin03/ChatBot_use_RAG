import faiss
import numpy as np
import json
import os
import speech_recognition as sr
from gtts import gTTS
from sentence_transformers import SentenceTransformer
from gradio_client import Client
from sklearn.metrics.pairwise import cosine_similarity


def initialize_components():
    """Khởi tạo các thành phần: FAISS, Sentence Transformer, tài liệu và kết nối Gradio Client."""
    # Load Sentence Transformer model
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Load tài liệu
    with open("vectorstores/documents.json", 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # Load FAISS index
    vector_db_path = "vectorstores/db_faiss"
    index = faiss.read_index(vector_db_path)

    # Kết nối Gradio Client (thay đổi API nếu cần)
    client = Client("yuntian-deng/ChatGPT")

    return sentence_model, documents, index, client


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


def search_and_predict(user_input, chat_context, model, index, documents, client):
    """Tìm kiếm tài liệu bằng FAISS + ranking cosine similarity, rồi gửi truy vấn đến ChatGPT."""
    chat_context.append(f"User: {user_input}")

    # Tạo embedding cho truy vấn
    input_embedding = model.encode([user_input]).astype('float32')

    # Tìm top-k tài liệu từ FAISS
    k = 10
    D, I = index.search(input_embedding, k)

    # Lấy tài liệu liên quan (lọc những index hợp lệ)
    retrieved_docs = [(i, documents[i]) for i in I[0] if i < len(documents)]

    if not retrieved_docs:
        combined_info = "Không tìm thấy tài liệu nào."
    else:
        # Tính điểm tương đồng cosine giữa truy vấn và tài liệu
        retrieved_embeddings = np.array([model.encode([doc]).astype('float32')[0] for _, doc in retrieved_docs])
        similarity_scores = cosine_similarity(input_embedding, retrieved_embeddings)[0]

        # Sắp xếp tài liệu theo độ tương đồng giảm dần
        sorted_docs = sorted(zip(similarity_scores, retrieved_docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, (_, doc) in sorted_docs[:5]]  # Chọn 5 tài liệu phù hợp nhất

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


def recognize_speech_from_microphone():
    """Nhận diện giọng nói từ micro và chuyển thành text."""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("🎤 Đang lắng nghe...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("🔍 Đang nhận dạng giọng nói...")
        text = recognizer.recognize_google(audio, language='vi-VN')
        print(f"🗣 Người dùng nói: {text}")
        return text
    except sr.UnknownValueError:
        print("⚠ Không thể nhận diện giọng nói.")
        return None
    except sr.RequestError:
        print("❌ Lỗi kết nối với dịch vụ nhận diện giọng nói.")
        return None


def main():
    """Chương trình chính: Hỗ trợ text + giọng nói."""
    # Khởi tạo các thành phần
    sentence_model, documents, index, client = initialize_components()

    # Danh sách lưu hội thoại
    chat_context = []

    sound_mode = False  # Mặc định ở chế độ nhập text

    print("🤖 Xin chào! Tôi có thể giúp gì cho bạn?")

    while True:
        if sound_mode:
            print("🎙 Nói điều gì đó...")
            user_input = recognize_speech_from_microphone()
            if user_input is None:
                continue
        else:
            user_input = input("📝 Bạn: ")

        if user_input.lower() in ["exit", "quit", "thoát"]:
            print("👋 Tạm biệt!")
            break

        elif user_input.lower() == "mic":
            print("🔊 Chế độ giọng nói đã được kích hoạt!")
            sound_mode = True
            continue

        elif user_input.lower() == "text":
            print("⌨ Chế độ văn bản đã được kích hoạt.")
            sound_mode = False
            continue

        # Xử lý câu hỏi của người dùng
        if user_input:
            response = search_and_predict(user_input, chat_context, sentence_model, index, documents, client)
            print("🤖 ChatBot:", response)


if __name__ == "__main__":
    main()
