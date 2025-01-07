// Function to reveal the chatbox when the photo is clicked
function revealChat() {
    const chatbox = document.getElementById('chatbox');
    const photoContainer = document.getElementById('photoContainer');

    if (!chatbox.classList.contains('slide-down')) {
        photoContainer.classList.add('slide-down'); // Slide image down
        chatbox.style.display = 'block'; // Show chatbox
        chatbox.classList.add('slide-down'); // Add animation class to chatbox

        // Hide photo container after animation ends
        photoContainer.addEventListener('animationend', () => {
            photoContainer.style.display = 'none';
        }, { once: true });

        chatbox.addEventListener('animationend', () => {
            chatbox.classList.remove('slide-down');
            chatbox.style.transform = 'translateY(300)';
            chatbox.style.position = 'fixed';
            chatbox.style.bottom = '55px';
            chatbox.style.left = '0';
            chatbox.style.width = '100%';
        }, { once: true });
    }
}

// Function to send a message
async function sendMessage() {
    const userInput = document.getElementById('userInput').value;
    if (userInput.trim() === "") return; // Don't send if the input is empty

    const chatDiv = document.getElementById('chat');

    // Append user's message to the chat
    chatDiv.innerHTML += `<div class="message user"><p><b>You:</b> ${userInput}</p></div>`;

    // Scroll to the latest message
    chatDiv.scrollTop = chatDiv.scrollHeight;

    // Clear the input field
    document.getElementById('userInput').value = "";

    // Create a container for the bot's response
    const botResponseContainer = document.createElement('div');
    botResponseContainer.classList.add('message', 'bot');
    botResponseContainer.innerHTML = `
        <img src="./client/Images/cat-gpt-logo.png" alt="Bot" class="bot-icon">
        <p><b>Bot is typing...</b></p>
    `;
    chatDiv.appendChild(botResponseContainer);

    try {
        // Fetch the streaming response
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userInput }),
        });

        if (response.body) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let botResponse = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                // Decode and parse the chunk
                const chunk = decoder.decode(value);
                const parsedChunk = JSON.parse(chunk);

                // Update the bot's response
                botResponse = parsedChunk.response;
                botResponseContainer.innerHTML = `
                    <img src="./client/Images/cat-gpt-logo.png" alt="Bot" class="bot-icon">
                    <p>${botResponse}</p>
                `;
            }
        } else {
            // Handle non-streaming response
            const data = await response.json();
            botResponseContainer.innerHTML = `
                <img src="./client/Images/cat-gpt-logo.png" alt="Bot" class="bot-icon">
                <p>${data.response}</p>
            `;
        }
    } catch (error) {
        console.error("Error fetching chat response:", error);

        // Display error message
        botResponseContainer.innerHTML = `
            <img src="./client/Images/cat-gpt-logo.png" alt="Bot" class="bot-icon">
            <p><b>Bot:</b> Sorry, I encountered an error.</p>
        `;
    } finally {
        // Ensure the scroll is updated
        chatDiv.scrollTop = chatDiv.scrollHeight;
    }
}

document.getElementById('userInput').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        sendMessage();
    }
});
