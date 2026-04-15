import os
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END

load_dotenv()

# --- 1. DEFINE THE STATE ---
# This dictionary is passed between every node in our graph.
class GraphState(TypedDict):
    question: str
    context: List[str]
    answer: str

# --- 2. DEFINE THE NODES (The "Workers") ---

def retrieve_node(state: GraphState):
    """Finds relevant chunks from ChromaDB."""
    print("🔍 [Node: Retrieve] Searching the database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./data/chroma_db", embedding_function=embeddings)
    
    docs = vectorstore.similarity_search(state["question"], k=3)
    context = [doc.page_content for doc in docs]
    return {"context": context}

def generate_node(state: GraphState):
    """Generates an answer using the retrieved context."""
    print("🧠 [Node: Generate] Drafting answer...")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    context_str = "\n\n".join(state["context"])
    prompt = f"""You are a helpful assistant. Use the context below to answer the user's question.
    Context: {context_str}
    Question: {state['question']}
    Answer:"""
    
    response = llm.invoke(prompt)
    return {"answer": response.content}

# --- 3. BUILD THE GRAPH ---

workflow = StateGraph(GraphState)

# Add our workers (nodes)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# Connect them (edges)
workflow.set_entry_point("retrieve")      # Start here
workflow.add_edge("retrieve", "generate") # Then go here
workflow.add_edge("generate", END)        # Then stop

# Compile the graph
app = workflow.compile()

if __name__ == "__main__":
    inputs = {"question": "What is LangGraph and how does it help with RAG?"}
    for output in app.stream(inputs):
        print(output)