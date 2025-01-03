from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

# Directory setup for static files
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/client", StaticFiles(directory=BASE_DIR / "client"), name="client")

# Serve HTML (index.html as a static file)
@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "client" / "index.html")

# Chat endpoint
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    response = f"am primit {user_message}"  # Temporary response for testing
    return JSONResponse({"response": response})
