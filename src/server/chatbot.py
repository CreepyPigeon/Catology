import string

import aiml

# Initialize AIML Kernel globally to avoid reloading for each request
kernel = aiml.Kernel()
kernel.learn("chat_rules.aiml")

def process_message(message: str) -> str:
    # Normalize the message
    message = message.lower().strip()
    message = message.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation

    # Get the bot's response
    response = kernel.respond(message)

    if response:
        return response
    else:
        # aici il b?gam pe gepeto la ac?iune
        return "I am sorry but i cannot answer that yet"
