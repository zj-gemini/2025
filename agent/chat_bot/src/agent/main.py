import sys
import pathlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Add project root to path to allow running as a script from any location.
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from src.agent.conversation import Message
from src.agent.planner import run as run_planner
from src.db import init_db, get_all_tickets

app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # The origin of the frontend app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/tickets")
def read_tickets():
    return get_all_tickets()


@app.post("/chat")
async def chat(messages: List[Message]):
    """
    Receives the entire conversation history and returns a response.
    """
    if not messages:
        return {"response": "No messages received. Please send a message."}
    response_text = run_planner(messages)
    return {"response": response_text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.agent.main:app", host="0.0.0.0", port=8000, reload=True)
