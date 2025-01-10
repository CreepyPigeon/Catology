import json
import asyncio
import random

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from starlette.responses import StreamingResponse
from src.server.chatbot import process_message

app = FastAPI()

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
    bot_response, response_flag = process_message(user_message)

    # Return the response as a stream of JSON objects
    return StreamingResponse(
        generate_response_json(bot_response, response_flag), media_type="application/json"
    )


if __name__ == "__main__":
    # Run the FastAPI app with uvicorn from the main function
    uvicorn.run(app, host="localhost", port=8001)
