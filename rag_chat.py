from config import DATA_FILE
from data_loader import load_dataset
from embedder import build_vector_db
from retriever import retrieve_relevant_chunks
from generator import generate_response
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys

app = FastAPI()

# Add CORS middleware so your frontend (e.g. SignalR app) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend origin for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

vector_db = build_vector_db(load_dataset(DATA_FILE))

@app.post("/chat")
async def chat(req: QueryRequest):
    top_chunks = retrieve_relevant_chunks(req.query, vector_db)
    context = [chunk for chunk, _ in top_chunks]
    response = generate_response(req.query, context)
    return {"response": response, "chunks": context}

@app.get("/")
async def root():
    return {
        "message": "🤖 RAG Chatbot API is running!",
        "usage": "Send a POST request to /chat with a JSON body like { 'query': 'Your question here' }"
    }

def run_cli():
    print("\n🧠 RAG Chatbot Ready (CLI Mode). Type 'exit' to quit.\n")
    try:
        while True:
            query = input("You: ")
            if query.strip().lower() in ("exit", "quit"):
                print("👋 Exiting chatbot.")
                break
            top_chunks = retrieve_relevant_chunks(query, vector_db)
            # print("\n📚 Retrieved Knowledge:")
            for chunk, score in top_chunks:
                print(f" - (similarity: {score:.2f}) {chunk}")
            response = generate_response(query, [chunk for chunk, _ in top_chunks])
            print("\n🤖 Chatbot response:")
            print(response)
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Interrupted. Exiting chatbot.")

def main():
    print("\nEnter 'cli' to run in terminal or 'unicorn' to run as a server:")
    mode = input("Mode: ").strip().lower()
    if mode == "cli":
        run_cli()
    elif mode == "unicorn":
        print("🚀 Launching Unicorn server at http://0.0.0.0:8000 ...")
        uvicorn.run("rag_chat:app", host="0.0.0.0", port=8000, reload=False)
    else:
        print("❌ Invalid option. Please enter 'cli' or 'unicorn'.")

if __name__ == "__main__":
    main()
