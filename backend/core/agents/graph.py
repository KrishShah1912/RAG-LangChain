__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import pickle
from typing import List, TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain / LangGraph Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch

# Day 11/12 Components
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from backend.core.ingestion.hybrid_retriever import create_hybrid_retriever

load_dotenv()

# --- 1. STATE DEFINITION ---
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: List[str]
    sources: List[str] 
    answer: str        # Used for grading status ('yes'/'no')
    retry_count: int

# --- 2. LLM SETUP ---
fast_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
smart_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# --- 3. HELPER ---
def get_text(message):
    if isinstance(message, tuple): return message[1]
    if hasattr(message, "content"): return message.content
    return str(message)

# --- 4. NODES ---

def retrieve_node(state: GraphState):
    """Hybrid Retrieval + Cohere Reranking with Cloud Path Fixes."""
    last_message = get_text(state["messages"][-1])
    print(f"🔍 [Node: Retrieve] Query: {last_message}")
    
    # CLOUD PATH FIX: Use absolute paths relative to this file
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up to project root from backend/core/agents/
    base_dir = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
    db_path = os.path.join(base_dir, "data", "chroma_db")
    pickle_path = os.path.join(base_dir, "data", "raw_documents.pkl")

    # SAFETY CHECK: If files are missing (not pushed to Git), skip to search
    if not os.path.exists(pickle_path) or not os.path.exists(db_path):
        print("⚠️ Local data files missing. Skipping to search routing.")
        return {"context": [], "sources": [], "retry_count": 0}

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        with open(pickle_path, "rb") as f:
            docs = pickle.load(f)
            
        base_retriever = create_hybrid_retriever(docs, vectorstore)

        # Apply Cohere Reranker
        compressor = CohereRerank(model="rerank-english-v3.0", top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        
        results = compression_retriever.invoke(last_message)
        
        return {
            "context": [d.page_content for d in results],
            "sources": [d.metadata.get("source", "Local Docs") for d in results],
            "retry_count": 0
        }
    except Exception as e:
        print(f"❌ Retrieval Error: {e}")
        return {"context": [], "sources": [], "retry_count": 0}

def grade_documents_node(state: GraphState):
    """Strictly evaluates if local context answers the question."""
    print("⚖️ [Node: Grade Documents]")
    if not state.get("context") or len(state["context"]) == 0: 
        return {"answer": "no"}
    
    last_message = get_text(state["messages"][-1])

    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    llm_with_tool = fast_llm.with_structured_output(Grade)
    prompt = ChatPromptTemplate.from_template(
        "You are a strict grader. Does the context provide a SPECIFIC answer to the question?\n"
        "Question: {question}\nContext: {context}\nReturn 'yes' or 'no'."
    )
    
    chain = prompt | llm_with_tool
    try:
        score = chain.invoke({"question": last_message, "context": state["context"][0]})
        print(f"--- GRADER SCORE: {score.binary_score.upper()} ---")
        return {"answer": score.binary_score.lower()}
    except:
        return {"answer": "no"}

def web_search_node(state: GraphState):
    """Fallback to Tavily Web Search (Updated for 2026 Package)."""
    print("🌐 [Node: Web Search]")
    last_message = get_text(state["messages"][-1])
    
    # 2026 Renaming Fix: use TavilySearch class
    search_tool = TavilySearch(max_results=3)
    
    try:
        results = search_tool.invoke({"query": last_message})
        return {
            "context": [r["content"] for r in results],
            "sources": [r["url"] for r in results],
            "retry_count": 1
        }
    except Exception as e:
        print(f"❌ Web Search Error: {e}")
        return {"context": [f"Search Error: {e}"], "sources": [], "retry_count": 1}

def generate_node(state: GraphState):
    """Synthesizes answer with memory and citations."""
    print("🧠 [Node: Generate Answer]")
    
    unique_sources = sorted(list(set(state.get("sources", []))))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert assistant. Use the provided context to answer. If you use Web Search data, be specific."),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    last_user_msg = get_text(state["messages"][-1])
    
    chain = prompt | smart_llm
    response = chain.invoke({
        "messages": state["messages"],
        "context": "\n\n".join(state["context"]), 
        "question": last_user_msg
    })
    
    full_content = response.content
    if unique_sources:
        full_content += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in unique_sources])

    return {
        "answer": full_content,
        "messages": [AIMessage(content=full_content)]
    }

# --- 5. GRAPH ROUTING ---

def decide_to_generate(state):
    if state.get("answer") == "yes":
        return "generate"
    return "web_search"

# --- 6. CONSTRUCTION ---

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade", 
    decide_to_generate, 
    {"generate": "generate", "web_search": "web_search"}
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile(checkpointer=MemorySaver())