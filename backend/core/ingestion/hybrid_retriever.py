# Updated for LangChain 2026 v1 Modular Structure
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma

def create_hybrid_retriever(documents, vectorstore):
    """
    Combines BM25 (Keyword) and Chroma (Vector) for hybrid search.
    """
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3
    
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever