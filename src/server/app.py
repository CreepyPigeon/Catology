import asyncio
import json

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
async def generate_response_json(response: str):
    """Generate a JSON response word by word."""
    words = response.split()  # Split the response into words
    partial_response = ""
    for word in words:
        partial_response += word + " "  # Append the current word
        await asyncio.sleep(0.25)  # Simulate delay
        yield json.dumps({"response": partial_response.strip()}) + "\n"

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    bot_response = process_message(user_message)

    # Return the response as a stream of JSON objects
    return StreamingResponse(
        generate_response_json(bot_response), media_type="application/json"
    )


if __name__ == "__main__":
    # Run the FastAPI app with uvicorn from the main function
    uvicorn.run(app, host="localhost", port=8001)
