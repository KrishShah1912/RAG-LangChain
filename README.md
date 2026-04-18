# Agentic-RAG: Enterprise Self-Corrective AI Research Engine

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph_v0.2-orange.svg)](https://github.com/langchain-ai/langgraph)
[![LLM-Groq](https://img.shields.io/badge/LLM-Groq/Llama3.3-green.svg)](https://groq.com/)
[![Search-Tavily](https://img.shields.io/badge/Search-Tavily-blue.svg)](https://tavily.com/)

An advanced **Corrective Retrieval-Augmented Generation (CRAG)** system. Unlike traditional RAG, this engine treats retrieval as an iterative reasoning task—validating, reranking, and searching the live web to ensure high-fidelity, zero-hallucination outputs.

---

## System Architecture and Workflow

The system is orchestrated as a **Stateful Micro-Agentic Graph**. Instead of a linear pipeline, the agent evaluates data quality at every logical junction, deciding whether to trust local documents or escalate to a live web search.

### The Intelligent Reasoning Loop

* **Hybrid Retrieval:** Simultaneous keyword (BM25) and semantic (ChromaDB) search for maximum recall.
* **Document Grading:** A specialized LLM node evaluates the relevance of each retrieved chunk against the user's intent.
* **Reranking:** Relevant documents are re-ordered using **Cohere Rerank v3** to ensure the most critical context is at the top of the prompt.
* **Decision Logic:** * **Context Sufficient?** Routes directly to Generation.
    * **Context Lacking/Irrelevant?** Triggers Query Transformation and **Tavily Web Search**.
* **Verified Generation:** The final answer is synthesized using only verified context, providing transparent citations and source URLs.

---

## Tech Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Orchestration** | **LangGraph** | State machine management and conditional routing. |
| **Inference** | **Groq (Llama 3.3-70B)** | High-speed LLM for grading, rewriting, and synthesis. |
| **Search Engine** | **Tavily AI** | Optimized search for real-time web grounding. |
| **Reranking** | **Cohere AI** | Contextual reranking to prioritize the highest quality data. |
| **Vector Database** | **ChromaDB** | Local persistence for high-dimensional document embeddings. |
| **Embeddings** | **HuggingFace** | `all-MiniLM-L6-v2` for efficient, local text vectorization. |
| **Frontend UI** | **Streamlit** | Professional dashboard with real-time process streaming. |

---

## Key Features

* **Self-Correction:** Automatically identifies and filters irrelevant information to prevent hallucinations.
* **Agentic Search:** Autonomously executes web searches when local knowledge is insufficient or outdated.
* **Conversational Memory:** Uses thread-safe persistence to maintain multi-turn dialogue context.
* **Hybrid Retrieval Logic:** Combines the precision of keyword search with the depth of semantic search.
* **Source Attribution:** Generates responses with specific citations and verifiable source links.

---

## Getting Started

### 1. Installation

```bash
git clone [https://github.com/YOUR_USERNAME/Agentic-RAG.git](https://github.com/YOUR_USERNAME/Agentic-RAG.git)
cd Agentic-RAG
pip install -r requirements.txt
```

### 2. Environment Setup
```bash 
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key
COHERE_API_KEY=your_cohere_key
```

### 3. Ingest your documents
```bash 
python backend/core/ingestion/ingest.py
```

### 4. Launch the Dashboard
```bash 
streamlit run frontend/app.py
```

---

## Project Structure

```text
├── backend/
│   ├── core/
│   │   ├── agents/      # LangGraph node and graph definitions
│   │   ├── ingestion/   # Document processing and vectorization
│   │   └── tools/       # Custom retrieval and web search tools
├── frontend/
│   └── app.py           # Streamlit UI and event stream handling
├── requirements.txt     # Production-ready dependencies
└── README.md            # Project documentation
```

---