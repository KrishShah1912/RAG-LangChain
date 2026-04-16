import os
from dotenv import load_dotenv
from src.agents.graph import app

load_dotenv()

def run_agent():
    print("--- 🤖 LangChain RAG Agent ---")
    print("Type 'exit' or 'quit' to stop.")
    
    # Define a thread_id for this session's memory
    config = {"configurable": {"thread_id": "main_session_1"}}

    while True:
        user_query = input("\n👤 You: ")
        
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        inputs = {"question": user_query}
        
        # Run the graph
        for event in app.stream(inputs, config=config, stream_mode="updates"):
            for node, values in event.items():
                # We only want to print the final answer to the user, 
                # but we'll print node names for debugging.
                print(f"📍 [Processing: {node}]")
                if "answer" in values:
                    print(f"\n🤖 Agent: {values['answer']}")

if __name__ == "__main__":
    run_agent()