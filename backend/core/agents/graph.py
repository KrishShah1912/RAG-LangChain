import os
import pickle
import operator
from typing import List, TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults

# Day 11: Professional Reranking
from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from backend.core.ingestion.hybrid_retriever import create_hybrid_retriever

load_dotenv()

# --- 1. STATE DEFINITION ---
class GraphState(TypedDict):
    # Annotated with add_messages so the history appends rather than overwrites
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
    """Hybrid Retrieval + Cohere Reranking."""
    last_message = get_text(state["messages"][-1])
    print(f"🔍 [Node: Retrieve & Rerank] Query: {last_message}")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    db_path = os.path.join(base_dir, "data/chroma_db")
    pickle_path = os.path.join(base_dir, "data/raw_documents.pkl")

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

def grade_documents_node(state: GraphState):
    """Strictly evaluates if local context answers the question."""
    print("⚖️ [Node: Grade Documents]")
    if not state.get("context"): return {"answer": "no"}
    
    last_message = get_text(state["messages"][-1])

    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    llm_with_tool = fast_llm.with_structured_output(Grade)
    prompt = ChatPromptTemplate.from_template(
        "You are a strict grader. Does the context provide a SPECIFIC answer to the question?\n"
        "Question: {question}\nContext: {context}\nReturn 'yes' or 'no'."
    )
    
    chain = prompt | llm_with_tool
    score = chain.invoke({"question": last_message, "context": state["context"][0]})
    print(f"--- GRADER SCORE: {score.binary_score.upper()} ---")
    return {"answer": score.binary_score}

def web_search_node(state: GraphState):
    """Fallback to Tavily Web Search."""
    print("🌐 [Node: Web Search]")
    last_message = get_text(state["messages"][-1])
    search_tool = TavilySearchResults(k=3)
    
    try:
        results = search_tool.invoke({"query": last_message})
        return {
            "context": [r["content"] for r in results],
            "sources": [r["url"] for r in results],
            "retry_count": 1
        }
    except Exception as e:
        return {"context": [f"Search Error: {e}"], "sources": [], "retry_count": 1}

def generate_node(state: GraphState):
    """Synthesizes answer with memory and citations."""
    print("🧠 [Node: Generate Answer]")
    
    unique_sources = sorted(list(set(state.get("sources", []))))
    
    # We use ChatPromptTemplate with a placeholder for history
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
    
    # Append sources manually to the bottom of the message
    full_content = response.content
    if unique_sources:
        full_content += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in unique_sources])

    return {
        "answer": full_content,
        "messages": [AIMessage(content=full_content)]
    }

# --- 5. GRAPH ROUTING ---

def decide_to_generate(state):
    if state["answer"] == "yes":
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
workflow.add_conditional_edges("grade", decide_to_generate, {"generate": "generate", "web_search": "web_search"})
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile(checkpointer=MemorySaver())