import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def run_chat_query(question: str):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        persist_directory="./data/chroma_db", 
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    system_prompt = (
        "You are a helpful assistant for LangChain documentation. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say you don't know.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({"context": context, "question": question})
    return response.content

if __name__ == "__main__":
    query = "How do I create a simple RAG chain in LangChain?"
    print(f"❓ Query: {query}")
    print("-" * 30)
    print(f"🤖 AI: {run_chat_query(query)}")