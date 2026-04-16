import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup as Soup

load_dotenv()

def ingest_data():
    print("🌐 Starting Ingestion...")

    # 1. Load Data
    url = "https://python.langchain.com/docs/introduction/"
    loader = RecursiveUrlLoader(
        url=url, 
        max_depth=1, # Depth 1 is faster for testing
        extractor=lambda x: Soup(x, "html.parser").text
    )
    docs = loader.load()
    
    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # 3. Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./data/chroma_db"
    )

    # 4. Save for BM25
    os.makedirs("./data", exist_ok=True)
    with open("./data/raw_documents.pkl", "wb") as f:
        pickle.dump(split_docs, f)
    
    print("🚀 Ingestion Complete!")

if __name__ == "__main__":
    ingest_data()