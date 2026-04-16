import os
import pickle
from typing import List, TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Import the hybrid retriever helper
from backend.core.ingestion.hybrid_retriever import create_hybrid_retriever

load_dotenv()

# --- 1. STATE DEFINITION ---
class GraphState(TypedDict):
    question: str
    context: List[str]
    answer: str
    retry_count: int

# --- 2. LLM SETUP ---
# Use 'llama-3.3-70b-versatile' as the current stable 70B model on Groq
fast_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
smart_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# --- 3. NODES ---

def retrieve_node(state: GraphState):
    """
    Combines Vector and Keyword search to find relevant context.
    """
    print(f"🔍 [Node: Hybrid Retrieve] Query: {state['question']}")
    
    # Pathing logic to find the 'data' folder at the project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    db_path = os.path.join(base_dir, "data/chroma_db")
    pickle_path = os.path.join(base_dir, "data/raw_documents.pkl")

    # Guard rail in case ingestion hasn't run
    if not os.path.exists(db_path) or not os.path.exists(pickle_path):
        return {"answer": "❌ Error: Data files not found. Run ingest.py first!", "context": []}

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    with open(pickle_path, "rb") as f:
        docs = pickle.load(f)
        
    retriever = create_hybrid_retriever(docs, vectorstore)
    results = retriever.invoke(state["question"])
    
    return {
        "context": [d.page_content for d in results], 
        "retry_count": state.get("retry_count", 0)
    }

def grade_documents_node(state: GraphState):
    """
    Determines if the retrieved context is actually useful.
    """
    if not state.get("context"): 
        return {"answer": "no"}
    
    print("⚖️ [Node: Grade Documents]")
    
    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    llm_with_tool = fast_llm.with_structured_output(Grade)
    prompt = ChatPromptTemplate.from_template(
        "You are a grader. Check if the context is relevant to the question.\n"
        "Question: {question}\n"
        "Context: {context}\n"
        "Return 'yes' if relevant, 'no' if not."
    )
    
    chain = prompt | llm_with_tool
    score = chain.invoke({
        "question": state["question"], 
        "context": state["context"][0]
    })
    
    return {"answer": score.binary_score}

def rewrite_node(state: GraphState):
    """
    Rewrites the question if the initial search yielded no good results.
    """
    print("✍️ [Node: Rewrite Question]")
    
    prompt = ChatPromptTemplate.from_template(
        "The previous search for '{question}' was not specific enough. "
        "Rewrite this into a better technical search query."
    )
    
    chain = prompt | fast_llm
    new_q = chain.invoke({"question": state["question"]}).content
    
    return {
        "question": new_q, 
        "retry_count": state.get("retry_count", 0) + 1
    }

def generate_node(state: GraphState):
    """
    Generates the final human-readable answer.
    """
    # Don't try to generate if there's a file system error
    if "❌ Error" in state.get("answer", ""): 
        return state
        
    print("🧠 [Node: Generate Answer]")
    
    prompt = ChatPromptTemplate.from_template(
        "You are a technical expert. Answer the question based ONLY on the context below.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Final Answer:"
    )
    
    chain = prompt | smart_llm
    response = chain.invoke({
        "context": "\n\n".join(state["context"]), 
        "question": state["question"]
    })
    
    return {"answer": response.content}

# --- 4. CONDITIONAL LOGIC ---

def decide_to_generate(state: GraphState):
    # Go to generate if relevance is 'yes' OR we've already retried twice
    if state.get("answer") == "yes" or state.get("retry_count", 0) >= 2:
        return "generate"
    return "rewrite"

# --- 5. GRAPH CONSTRUCTION ---

workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)

# Define Connections
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

# Compile with In-Memory Checkpointer for session persistence
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)