document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    // Predefined calming responses based on keywords
    const responses = {
        crisis: [
            "I'm really sorry you're feeling this way. Your life matters, and there are people who care. Please reach out to a hotline right now—they're trained to help.",
            "Take a deep breath. You're not alone in this. Let's talk about what's going on. Remember, help is available 24/7.",
            "It's okay to feel overwhelmed, but ending things isn't the answer. You've survived tough times before—let's find a way through this together."
        ],
        general: [
            "I'm here to listen. What's on your mind?",
            "Take it one step at a time. You're stronger than you think.",
            "How can I support you right now?"
        ]
    };

    // Keywords to detect crisis (case-insensitive)
    const crisisKeywords = ['suicide', 'kill myself', 'end it', 'hopeless', 'depressed', 'want to die', 'crisis'];

    // Function to add a message to the chat box
    function addMessage(text, isUser) {
        const message = document.createElement('div');
        message.classList.add('message');
        message.classList.add(isUser ? 'user-message' : 'bot-message');
        message.textContent = text;
        chatBox.appendChild(message);
        chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
    }

    // Function to get a bot response
    function getBotResponse(userMessage) {
        const lowerMessage = userMessage.toLowerCase();
        // Check for crisis keywords
        if (crisisKeywords.some(keyword => lowerMessage.includes(keyword))) {
            // Randomly pick a crisis response
            const randomIndex = Math.floor(Math.random() * responses.crisis.length);
            return responses.crisis[randomIndex] + ' Call Sumithrayo at 011-269-6666 if in Sri Lanka.';
        } else {
            // Random general response
            const randomIndex = Math.floor(Math.random() * responses.general.length);
            return responses.general[randomIndex];
        }
    }

    // Event listener for send button
    sendBtn.addEventListener('click', () => {
        const userMessage = userInput.value.trim();
        if (userMessage) {
            addMessage(userMessage, true);
            const botResponse = getBotResponse(userMessage);
            addMessage(botResponse, false);
            userInput.value = ''; // Clear input
        }
    });

    // Allow pressing Enter to send
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendBtn.click();
        }
    });

    // Initial bot greeting
    addMessage("Hi, I'm here to help calm you down. What's going on?", false);
});