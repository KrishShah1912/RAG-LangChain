import os
from typing import List, TypedDict, Literal
from dotenv import load_dotenv

# LangChain & Groq Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# LangGraph Imports (2026 Syntax)
from langgraph.graph import StateGraph, START, END

load_dotenv()

# --- 1. DEFINE THE STATE ---
class GraphState(TypedDict):
    question: str
    context: List[str]
    answer: str
    retry_count: int
    is_relevant: str # 'yes' or 'no'

# --- 2. DEFINE THE NODES ---

def retrieve_node(state: GraphState):
    """Searches the local ChromaDB."""
    print("\n🔍 [Node: Retrieve] Searching vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./data/chroma_db", 
        embedding_function=embeddings
    )
    
    # Perform search
    docs = vectorstore.similarity_search(state["question"], k=3)
    context = [doc.page_content for doc in docs]
    
    return {
        "context": context, 
        "retry_count": state.get("retry_count", 0)
    }

def grade_documents_node(state: GraphState):
    """Filters out irrelevant data."""
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
    """Improves the question if the first search was bad."""
    print("✍️ [Node: Rewrite] Optimizing search query...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    prompt = f"The previous search for '{state['question']}' yielded no relevant results. Rewrite this to be a more technical search query."
    new_query = llm.invoke(prompt).content
    
    return {
        "question": new_query, 
        "retry_count": state["retry_count"] + 1
    }

def generate_node(state: GraphState):
    """Generates the final human-like answer."""
    print("🧠 [Node: Generate] Drafting final response...")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    context_text = "\n\n".join(state["context"])
    prompt = f"""Use the following context to answer the question. 
    Context: {context_text}
    Question: {state['question']}
    Answer:"""
    
    response = llm.invoke(prompt)
    return {"answer": response.content}

# --- 3. CONDITIONAL ROUTING ---

def decide_to_generate(state: GraphState) -> Literal["generate", "rewrite"]:
    """Routing logic: Should we give up, rewrite, or answer?"""
    if state["is_relevant"] == "yes" or state["retry_count"] >= 2:
        return "generate"
    return "rewrite"

# --- 4. ASSEMBLE THE GRAPH ---

workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("generate", generate_node)

# Build Edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")

# Add the Router
workflow.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

# Loop back from rewrite to retrieve
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# --- 5. EXECUTION BLOCK ---

if __name__ == "__main__":
    print("\n🚀 Day 5: Corrective RAG Agent Online")
    user_input = {"question": "How do I create a LangChain retriever?"}
    
    # Stream the events so we see every node as it happens
    for event in app.stream(user_input, stream_mode="updates"):
        for node_name, state_update in event.items():
            print(f"🏁 Finished Node: {node_name}")
            if "answer" in state_update:
                print(f"\n🤖 FINAL ANSWER:\n{state_update['answer']}")