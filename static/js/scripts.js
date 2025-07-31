let chat_context = [];
let conversationId = null;

const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = "vi-VN";  
recognition.continuous = false; 
recognition.interimResults = false;  

const micBtn = document.getElementById("mic-btn");
const alertBox = document.getElementById("listening-alert");

// Bật nhận diện giọng nói khi nhấn vào nút
micBtn.addEventListener("mousedown", () => {
    recognition.start();
    alertBox.style.display = "block";
});

// Tắt nhận diện khi thả nút
micBtn.addEventListener("mouseup", stopVoice);
micBtn.addEventListener("mouseleave", stopVoice);

function stopVoice() {
    recognition.stop();
    alertBox.style.display = "none";
}

recognition.onresult = function(event) {
    const transcript = event.results[0][0].transcript;
    document.getElementById("user-input").value = transcript;
    toggleButtons();
    sendMessage();
};

recognition.onerror = function(event) {
    console.error("Lỗi nhận diện giọng nói:", event.error);
};

function toggleButtons() {
    const userInput = document.getElementById("user-input").value.trim();
    document.getElementById("send-btn").style.display = userInput ? "block" : "none";
    document.getElementById("mic-btn").style.display = userInput ? "none" : "block";
}

function handleKeyPress(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function sendMessage() {
    const inputField = document.getElementById("user-input");
    const message = inputField.value.trim();
    if (message === "") return;

    const chatBox = document.getElementById("chat-box");
    
    const userMsg = document.createElement("div");
    userMsg.classList.add("user-msg");
    userMsg.textContent = message;
    chatBox.appendChild(userMsg);
    
    inputField.value = "";
    toggleButtons();
    chatBox.scrollTop = chatBox.scrollHeight;
    
    setTimeout(() => {
        const typingIndicator = document.createElement("div");
        typingIndicator.classList.add("bot-msg");
        typingIndicator.classList.add("typing");
        chatBox.appendChild(typingIndicator);
        chatBox.scrollTop = chatBox.scrollHeight;

        setTimeout(() => {
            fetch('/get_response', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    user_input: message, 
                    chat_context: chat_context, 
                    conversation_id: conversationId 
                })
            })
            .then(response => response.json())
            .then(data => {
                chatBox.removeChild(typingIndicator);
                const botReply = document.createElement("div");
                botReply.classList.add("bot-msg");
                botReply.textContent = `🤖: ${data.response}`;
                chatBox.appendChild(botReply);
                chatBox.scrollTop = chatBox.scrollHeight;
                chat_context = data.chat_context;
                if (data.conversation_id) {
                    conversationId = data.conversation_id;
                }
            })
            .catch(error => console.error('Error:', error));
        }, 2000);
    }, 1000);
}
document.getElementById("sidebar").classList.remove("active");
const toggleBtn = document.getElementById("open-sidebar");
  const sidebar = document.getElementById("sidebar");

  toggleBtn.addEventListener("click", () => {
    sidebar.classList.toggle("active");
  });

const newChatBtn = document.getElementById("new-chat");
const chatBox = document.getElementById("chat-box");

newChatBtn.addEventListener("click", () => {
    while (chatBox.firstChild) {
        chatBox.removeChild(chatBox.firstChild);
    }

    const intro = document.createElement("h1");
    intro.id = "intro";
    intro.style.textAlign = "center";
    intro.textContent = "Tôi có thể giúp gì cho bạn?";
    chatBox.appendChild(intro);

    conversationId = null;     
    chat_context = [];         
});


document.getElementById("logout-btn").addEventListener("click", function () {
    window.location.href = "/logout";
});


function loadConversation(id) {
    conversationId = id;  

    fetch(`/get_conversation/${id}`)
        .then(response => response.json())
        .then(data => {
            const chatBox = document.getElementById("chat-box");
            chatBox.innerHTML = "";

            data.messages.forEach(msg => {
                const msgDiv = document.createElement("div");
                msgDiv.classList.add(msg.sender === "User" ? "user-msg" : "bot-msg");
                msgDiv.textContent = msg.message;
                chatBox.appendChild(msgDiv);
            });

            chatBox.scrollTop = chatBox.scrollHeight;

            
            if (data.chat_context) {
                chat_context = data.chat_context;
            }
        });
}

document.addEventListener("click", function (e) {
    if (e.target.classList.contains("btn-Delete")) {
        e.stopPropagation();
        const conversationId = e.target.getAttribute("data-id");

        if (confirm("Bạn có chắc chắn muốn xóa cuộc hội thoại này?")) {
            fetch(`/delete_conversation/${conversationId}`, {
                method: 'DELETE'
            })
            .then(response => {
                if (response.ok) {
                    e.target.closest("li").remove();

                    while (chatBox.firstChild) {
                        chatBox.removeChild(chatBox.firstChild);
                    }

                    const intro = document.createElement("h1");
                    intro.id = "intro";
                    intro.style.textAlign = "center";
                    intro.textContent = "Tôi có thể giúp gì cho bạn?";
                    chatBox.appendChild(intro);

                    conversationId = null;
                    chat_context = [];

                } else {
                    alert("Xóa thất bại!");
                }
            })
            .catch(error => {
                console.error("Lỗi khi xóa:", error);
            });
        }
    }
});
