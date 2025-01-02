import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

PROJECT_DIR = Path(__file__).resolve().parent.parent  # Point to the root directory of the project
CLIENT_DIR = PROJECT_DIR / "client"  # Path to the 'client' directory

# Mount the static files folder (client's static files)
app.mount("/static", StaticFiles(directory=CLIENT_DIR), name="static")

@app.get("/")
async def index():
    return FileResponse(CLIENT_DIR / "index.html")

# Chat endpoint
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    response = f"am primit {user_message}"  # Temporary response for testing
    return JSONResponse({"response": response})

if __name__ == '__main__':
    uvicorn.run(
        "server.app:app",  # The module and app to run
        host="127.0.0.1",  # Host address
        port=8000,  # Port number
        reload=True  # Enable reload for development
    )