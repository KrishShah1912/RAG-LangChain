# Agentic-RAG: Self-Corrective AI Research Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![LLM-Groq](https://img.shields.io/badge/LLM-Groq/Llama3.3-green.svg)](https://groq.com/)

An advanced **Corrective Retrieval-Augmented Generation (CRAG)** system that doesn't just find information—it evaluates it. Built with a modular graph architecture, this agent identifies irrelevant data, rewrites search queries for precision, and maintains long-term conversational memory.

---

## Project Overview
Unlike "Generic RAG" systems that often hallucinate when search results are poor, this agent implements a **Reasoning Loop**. It acts as its own quality controller, ensuring the context provided to the LLM is accurate and relevant before generating a response.

### Key Capabilities:
* **Self-Correction:** Automatically detects if retrieved documents are irrelevant to the user's query.
* **Query Refinement:** Rewrites vague questions into optimized technical search queries.
* **Persistent Memory:** Uses thread-safe checkpointers to remember conversation context.
* **Hybrid Model Logic:** Uses **Llama-3.1-8B** for high-speed reasoning (grading/rewriting) and **Llama-3.3-70B** for high-fidelity synthesis.

---

## System Architecture

The system is orchestrated as a **Stateful Graph**. Each node represents a specific logical step, and edges define the flow based on real-time evaluation.



### The Workflow:
1.  **Retrieve:** Performs a similarity search against a local **ChromaDB** vector store using **HuggingFace Embeddings**.
2.  **Grade:** A specialized node evaluates the "Relevance" of the documents. 
3.  **Decide:** * If relevant $\rightarrow$ Move to **Generate**.
    * If irrelevant/ambiguous $\rightarrow$ Move to **Rewrite**.
4.  **Rewrite:** The agent analyzes why the search failed and generates a more effective query to try again.
5.  **Generate:** The final response is drafted using the verified context, citing specific sources for transparency.

---

## The Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | **LangGraph** | Manages the state machine and conditional logic. |
| **LLM Inference** | **Groq (Llama 3.1 & 3.3)** | Ultra-fast inference for real-time agentic loops. |
| **Vector Database** | **ChromaDB** | High-performance local storage for document embeddings. |
| **Embeddings** | **HuggingFace (all-MiniLM-L6-v2)** | Efficient open-source text vectorization. |
| **Frontend UI** | **Streamlit** | Modern web interface with real-time process streaming. |
| **Data Ingestion**| **BeautifulSoup / LangChain** | Recursive web scraping and document chunking. |

---

## Getting Started

### 1. Clone & Install
```bash
git clone [https://github.com/YOUR_USERNAME/RAG-LangChain.git](https://github.com/YOUR_USERNAME/RAG-LangChain.git)
cd RAG-LangChain
pip install -r requirements.txt

### 2. Environment Setup
```bash 
GROQ_API_KEY=your_groq_key_here

### 3. Ingest Data
```bash 
python src/ingestion/ingest.py

### 4. Launch the Agent
```bash 
python -m streamlit run app.py

---

## Advanced Features Included

* Thread Persistence: Every user session is assigned a **thread_id**, allowing the agent to maintain state across multiple interactions.
* Visual Thinking: The UI provides a **Status Breadcrumb**, showing the user exactly which node the agent is currently processing.
* Source Attribution: The final generation node is prompted to cite specific indices from the retrieved context to minimize hallucinations.

---