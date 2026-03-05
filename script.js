document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    function addMessage(text, isUser) {
        const message = document.createElement('div');
        message.classList.add('message', isUser ? 'user-message' : 'bot-message');
        message.textContent = text;
        chatBox.appendChild(message);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function sendMessage() {
        const userMessage = userInput.value.trim();
        if (userMessage) {
            addMessage(userMessage, true);
            userInput.value = '';
            
            try {
                const response = await fetch('http://localhost:5000/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: userMessage })
                });
                const data = await response.json();
                addMessage(data.response, false);
            } catch (error) {
                addMessage('Error: Could not connect to AI.', false);
            }
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    addMessage("Hi! Ask me anything.", false);
});