import os
from typing import List, TypedDict, Literal
from dotenv import load_dotenv

# LangChain & Groq Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# --- 1. DEFINE THE STATE ---
class GraphState(TypedDict):
    question: str
    context: List[str]
    answer: str
    retry_count: int
    is_relevant: str 

# --- 2. DEFINE THE NODES ---

def retrieve_node(state: GraphState):
    """Searches the local ChromaDB."""
    print("\n🔍 [Node: Retrieve] Searching vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./data/chroma_db", 
        embedding_function=embeddings
    )
    
    docs = vectorstore.similarity_search(state["question"], k=3)
    context = [doc.page_content for doc in docs]
    
    # Initialize retry_count if it doesn't exist
    count = state.get("retry_count", 0)
    return {"context": context, "retry_count": count}

def grade_documents_node(state: GraphState):
    """Filters out irrelevant data using a fast LLM."""
    print("⚖️ [Node: Grade] Checking document relevance...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    doc_sample = state["context"][0] if state["context"] else "No context found"
    prompt = f"""Assess if the following context is relevant to the question.
    Question: {state['question']}
    Context: {doc_sample}
    Answer only 'yes' or 'no'."""
    
    response = llm.invoke(prompt).content.lower()
    is_relevant = "yes" if "yes" in response else "no"
    
    return {"is_relevant": is_relevant}

def rewrite_query_node(state: GraphState):
    """Improves the search query if initial results were poor."""
    print("✍️ [Node: Rewrite] Optimizing search query...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    prompt = f"The previous search for '{state['question']}' yielded no relevant results. Rewrite this to be a more technical search query for documentation."
    new_query = llm.invoke(prompt).content
    
    return {
        "question": new_query, 
        "retry_count": state["retry_count"] + 1
    }

def generate_node(state: GraphState):
    """Generates the final human-like answer using a high-reasoning LLM."""
    print("🧠 [Node: Generate] Drafting final response...")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    context_text = "\n\n".join(state["context"])
    prompt = f"""Use the following context to answer the question accurately. 
    Context: {context_text}
    Question: {state['question']}
    Answer:"""
    
    response = llm.invoke(prompt)
    return {"answer": response.content}

# --- 3. CONDITIONAL ROUTING ---

def decide_to_generate(state: GraphState) -> Literal["generate", "rewrite"]:
    """Determines the next step based on document relevance."""
    if state["is_relevant"] == "yes" or state["retry_count"] >= 2:
        return "generate"
    return "rewrite"

# --- 4. ASSEMBLE THE GRAPH WITH PERSISTENCE ---

workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("generate", generate_node)

# Build Edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")

# Add Routing Logic
workflow.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

# Loop back from rewrite
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

# NEW FOR DAY 6: Setup Memory Checkpointer
checkpointer = MemorySaver()

# Compile with Memory
app = workflow.compile(checkpointer=checkpointer)

# --- 5. EXECUTION BLOCK ---

if __name__ == "__main__":
    print("\n🚀 Persistent Corrective RAG Agent Online")
    
    # 'thread_id' acts as the save-file name for this conversation
    config = {"configurable": {"thread_id": "session_001"}}
    
    user_input = {"question": "What is the primary role of a LangChain retriever?"}
    
    # Use stream_mode="updates" to see each node finish in real-time
    for event in app.stream(user_input, config=config, stream_mode="updates"):
        for node_name, state_update in event.items():
            print(f"🏁 Node '{node_name}' finished.")
            if "answer" in state_update:
                print(f"\n🤖 AGENT RESPONSE:\n{state_update['answer']}")

    # Verification: Retrieve the state from memory
    final_state = app.get_state(config)
    print(f"\n💾 Persistence Check: Total Retries stored in memory: {final_state.values.get('retry_count')}")