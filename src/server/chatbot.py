import aiml
import string

# initialize AIML Kernel globally to avoid reloading for each request
kernel = aiml.Kernel()
kernel.learn("chat_rules.aiml")

def process_message(message: str):

    """
    :param message: the client's message
    :return: returns the response, alongside the origin (1 for aiml, 2 for pre-trained LLM)
    """

    original_message = message
    message = message.lower().strip()
    message = message.translate(str.maketrans('', '', string.punctuation))

    response = kernel.respond(message)
    print(f"Original: {original_message} | Processed: {message} | AIML Response: '{response}'")  # Debug

    if response:
        return response, 1
    else:
        return "I am sorry but i cannot answer that yet", 2

