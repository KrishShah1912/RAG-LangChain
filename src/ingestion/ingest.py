import os
from dotenv import load_dotenv
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Local & Free
from langchain_community.vectorstores import Chroma

load_dotenv()

def bs4_extractor(html: str) -> str:
    """Cleans up the HTML to just give the AI the good stuff."""
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text()

def run_ingestion():
    print("🚀 Starting Ingestion...")
    url = "https://python.langchain.com/docs/introduction/"
    loader = RecursiveUrlLoader(
        url=url, 
        max_depth=2, 
        extractor=bs4_extractor
    )
    raw_docs = loader.load()
    print(f"✅ Loaded {len(raw_docs)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(raw_docs)
    print(f"✂️ Split into {len(docs)} chunks")

    print("🧠 Embedding and saving to ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./data/chroma_db"
    )
    
    print("✨ Ingestion Completed! Data saved to ./data/chroma_db")

if __name__ == "__main__":
    run_ingestion()