import os
import pickle
import operator
from typing import List, TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults

# Import the hybrid retriever helper
from backend.core.ingestion.hybrid_retriever import create_hybrid_retriever

load_dotenv()

# --- HELPER FUNCTION ---
def get_text(message):
    """Safely extracts text whether the message is a tuple or a Message object."""
    if isinstance(message, tuple):
        return message[1]
    return message.content

# --- 1. STATE DEFINITION ---
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: List[str]
    answer: str
    retry_count: int

# --- 2. LLM SETUP ---
fast_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
smart_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# --- 3. NODES ---

def retrieve_node(state: GraphState):
    """Uses the last user message to trigger hybrid retrieval."""
    last_message = get_text(state["messages"][-1])
    print(f"🔍 [Node: Hybrid Retrieve] Query: {last_message}")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    db_path = os.path.join(base_dir, "data/chroma_db")
    pickle_path = os.path.join(base_dir, "data/raw_documents.pkl")

    if not os.path.exists(db_path) or not os.path.exists(pickle_path):
        return {"answer": "❌ Error: Data files not found.", "context": []}

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    with open(pickle_path, "rb") as f:
        docs = pickle.load(f)
        
    retriever = create_hybrid_retriever(docs, vectorstore)
    results = retriever.invoke(last_message)
    
    return {
        "context": [d.page_content for d in results], 
        "retry_count": state.get("retry_count", 0)
    }

def grade_documents_node(state: GraphState):
    """Checks if the retrieved context is relevant."""
    if not state.get("context"): 
        return {"answer": "no"}
    
    print("⚖️ [Node: Grade Documents]")
    last_message = get_text(state["messages"][-1])

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
        "question": last_message, 
        "context": state["context"][0]
    })
    
    print(f"--- GRADER SCORE: {score.binary_score.upper()} ---")
    return {"answer": score.binary_score}

def web_search_node(state: GraphState):
    """Fallback to web search and CLEAR old, useless context."""
    print("🌐 [Node: Web Search]")
    last_message = get_text(state["messages"][-1])
    
    search_tool = TavilySearchResults(k=3)
    try:
        search_results = search_tool.invoke({"query": last_message})
        # Extract content only
        content_list = [res["content"] for res in search_results]
        
        # CRITICAL: We return a NEW context list, effectively 
        # overwriting the "The context doesn't mention..." local data.
        return {
            "context": content_list, 
            "retry_count": 1 
        }
    except Exception as e:
        print(f"❌ Search Error: {e}")
        return {"context": ["Search failed. Use internal knowledge."], "retry_count": 1}

def generate_node(state: GraphState):
    """Generates the final answer with priority instructions."""
    print("🧠 [Node: Generate Answer]")
    last_message = get_text(state["messages"][-1])
    
    # We tell the LLM that if it sees a Harrison Chase or Ankush Gola, it's correct.
    prompt = ChatPromptTemplate.from_template(
        "You are an expert research assistant. You have been provided with specific context.\n\n"
        "GUIDELINE: If the context contains information (like founders or dates), provide a "
        "direct and complete answer. Do not say 'the context doesn't mention it' if the "
        "answer is present in the text below.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Final Answer:"
    )
    
    chain = prompt | smart_llm
    response = chain.invoke({
        "context": "\n\n".join(state["context"]), 
        "question": last_message
    })
    
    return {
        "answer": response.content,
        "messages": [AIMessage(content=response.content)]
    }

# --- 4. CONDITIONAL LOGIC ---

def decide_to_generate(state: GraphState):
    """Route to Web Search if local docs are irrelevant."""
    if state.get("answer") == "yes":
        return "generate"
    
    # If already tried web search once, just generate
    if state.get("retry_count", 0) >= 1:
        return "generate"
        
    return "web_search"

# --- 5. GRAPH CONSTRUCTION ---

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
    {
        "generate": "generate", 
        "web_search": "web_search"
    }
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Persistent Memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)