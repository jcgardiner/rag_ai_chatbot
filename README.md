# RAG Chatbot 🐱🔍

A specialized Retrieval-Augmented Generation chatbot that answers questions about cats using a comprehensive knowledge base of feline facts, powered by Llama 3. 

## Features ✨
- **Llama 3 Powered**: Uses quantized 1.2B parameter Llama-3-Instruct model
- **Cat Expert**: 200+ verified feline facts in knowledge base
- **Efficient Retrieval**: BGE embeddings with cosine similarity search
- **Dual Interface**: CLI and FastAPI server options
- **Transparent Sources**: Returns exact fact excerpts used for generation

## Tech Stack 🛠️
- **LLM**: `bartowski/Llama-3.2-1B-Instruct-GGUF`
- **Embeddings**: `CompendiumLabs/bge-base-en-v1.5-gguf`
- **Vector Search**: In-memory cosine similarity
- **Backend**: FastAPI + Uvicorn
- **Knowledge**: 200+ cat facts from curated dataset

## Installation 📦

### Prerequisites
- Python 3.10+
- pip
- 4GB RAM minimum (8GB recommended)

### Setup
```bash
git clone https://github.com/yourusername/cat-facts-rag.git
cd cat-facts-rag

# Install dependencies
pip install -r requirements.txt

# Download models (example using huggingface-hub)
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF --local-dir models/llama
huggingface-cli download CompendiumLabs/bge-base-en-v1.5-gguf --local-dir models/embedder

# Run the system
python rag_chat.py cli  # or 'server' for API mode
