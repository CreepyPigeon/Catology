import aiml
import json
import asyncio
import random
import string
import uvicorn
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse
from src.server.regex_commands import get_mapping_instance_addition, get_two_breeds
from src.server.prepared_responses import get_numerical_mapping, add_new_instance, compare_races

app = FastAPI()

kernel = aiml.Kernel()
kernel.learn("chat_rules.aiml")

# Directory setup for static files
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/client", StaticFiles(directory=BASE_DIR / "client"), name="client")

# Serve HTML (index.html as a static file)
@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "client" / "index.html")

# Chat endpoint
async def generate_response_json(response: str, flag):
    """Generate a JSON response word by word."""
    words = response.split()  # split the response into words

    timer = random.uniform(0.25, 0.5)
    partial_response = ""
    for word in words:
        partial_response += word + " "  # append the current word

        if flag == 1:
            await asyncio.sleep(timer)  # simulate delay
        yield json.dumps({"response": partial_response.strip()}) + "\n"


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    response, response_flag = process_message(user_message)

    # Return the response as a stream of JSON objects
    return StreamingResponse(
        generate_response_json(response, response_flag), media_type="application/json"
    )

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
        if "add a new instance in the cat dataset" in response.lower():
            mapping = get_mapping_instance_addition(message)
            # add_new_instance(mapping)  # Call your function to add the instance
            print(f"mapping = {mapping}")
            return "New row added successfully", 1
        return response, 1  # Regular AIML response
    else:  # If AIML did not provide a response
        if "add a new instance in the cat dataset" in message:
            mapping = get_mapping_instance_addition(message)
            numerical_mapping = get_numerical_mapping(mapping)

            if isinstance(numerical_mapping, int):
                return "I didn't quite understand that. Some columns and/or values were unidentifiable"
            else:
                print(f"initial mapping = {mapping}")
                print(f"numerical mapping = {numerical_mapping}")
                most_probable_class = add_new_instance(numerical_mapping)
                return f"New row added successfully. The cat is most probably of the following" \
                       f" race: {most_probable_class}", 1

        if "generate a natural language comparison between" in message:
            race1, race2 = get_two_breeds(message)
            if race1 != -1 and race2 != -1:
                print(f"Extracted races: {race1} and {race2}")
                additional_info = compare_races(race1, race2)
                base_str = f"The two races you've mentioned are {race1} and {race2}. "
                print(f"additional_info = {additional_info}")
                base_str += additional_info
                return base_str, 1

            else:
                return "Sorry, I couldn't extract the races from the message.", 2

        return "I am sorry but I cannot answer that yet", 2


if __name__ == "__main__":
    # Run the FastAPI app with uvicorn from the main function
    uvicorn.run(app, host="localhost", port=8001)
