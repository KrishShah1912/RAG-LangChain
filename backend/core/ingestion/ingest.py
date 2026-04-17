import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from bs4 import BeautifulSoup as Soup

load_dotenv()

def ingest_data():
    # 1. Load Data
    url = "https://python.langchain.com/docs/introduction/"
    loader = RecursiveUrlLoader(
        url=url, 
        max_depth=1, 
        extractor=lambda x: Soup(x, "html.parser").text
    )
    docs = loader.load()

    # 2. Embeddings (Used to detect semantic shifts)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Semantic Chunking
    # Breakpoint at 95th percentile of distance between sentences
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile"
    )
    
    split_docs = text_splitter.split_documents(docs)

    # 4. Save to ChromaDB (Wipe old one first)
    db_path = "./data/chroma_db"
    
    # Check if the folder exists and delete it
    if os.path.exists(db_path):
        print(f"🧹 Clearing old database at {db_path}...")
        import shutil
        shutil.rmtree(db_path)
        
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=db_path
    )

    # 5. Save Pickle for BM25
    os.makedirs("./data", exist_ok=True)
    with open("./data/raw_documents.pkl", "wb") as f:
        pickle.dump(split_docs, f)

if __name__ == "__main__":
    ingest_data()