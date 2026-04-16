![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?logo=langchain)
![LangGraph](https://img.shields.io/badge/LangGraph-Memory-blueviolet?logo=langchain)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange?logo=groq)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-lightblue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

# book-agent

A conversational AI agent that answers questions from your PDF documents using RAG (Retrieval-Augmented Generation). Drop any PDF into the `books/` folder and start asking questions the agent automatically indexes it and searches the right document based on your query.

## Architecture

```
User question
      │
      ▼
 Agent (LLM)
 decides which
 tool to use
      │
      ├──────────────────────────────────┐
      │                                  │
      ▼                                  ▼
search_document1              search_document2  ...
FAISS index                   FAISS index
document1.pdf                 document2.pdf
      │                                  │
      └──────────┬───────────────────────┘
                 │
                 ▼
         relevant chunks
         + page numbers
                 │
                 ▼
          final answer
```

Each PDF lives in `books/`, gets indexed into its own FAISS vector store, and is exposed to the agent as a dedicated search tool. The agent picks the right tool based on the question, retrieves the most relevant chunks, and returns a grounded answer with page references.

## How it works

```
books/
├── document1.pdf      →   search_document1  (tool)
├── document2.pdf      →   search_document2  (tool)
└── document3.pdf      →   search_document3  (tool)
```

## Stack

- **LangChain** — agent orchestration and RAG pipeline
- **LangGraph** — agent memory and conversation state
- **FAISS** — local vector store, no external service needed
- **HuggingFace Embeddings** — `all-MiniLM-L6-v2`, runs fully offline
- **Groq** — LLM inference (llama-3.3-70b or any supported model)

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Add your API key**

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
```

**3. Add PDFs**

```bash
mkdir books
cp your_document.pdf books/
```

**4. Index the PDFs**

```bash
python ingest.py
```

Run this once per new document. Existing indexes are skipped automatically.

**5. Start chatting**

```bash
python chat.py
```

## Adding a new document

Just drop the PDF into `books/` and run `ingest.py` again — it picks up only the new file and leaves existing indexes untouched. Restart `chat.py` to load the new tool.

## Project structure

```
book-agent/
├── books/           # your PDF files go here
├── indexes/         # FAISS indexes (auto-generated)
├── ingest.py        # PDF → chunks → FAISS index
├── tools.py         # auto-generates a search tool per book
├── agent.py         # agent setup with memory
├── chat.py          # terminal chat interface
├── requirements.txt
└── .env             # GROQ_API_KEY (not committed)
```

## Requirements

```
langchain
langchain-groq
langchain-community
langchain-huggingface
langchain-text-splitters
langgraph
faiss-cpu
sentence-transformers
pypdf
python-dotenv
```
