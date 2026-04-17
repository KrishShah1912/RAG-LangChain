import streamlit as st
import os
import sys
import uuid

# 1. SETUP: Ensure imports work across folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.core.agents.graph import app

st.set_page_config(page_title="RAG 2.0 Agent", layout="centered")
st.title("🤖 Agentic RAG 2.0")

# 2. SESSION STATE: Persistence Setup
# Unique ID for the current chat session to keep memory separate from other users
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Initialize message history for the UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. UI: Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. CHAT INPUT: Handling User Interaction
if prompt := st.chat_input("Ask a question (e.g., 'What is LangChain?')"):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response area
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        # Configure the graph with the thread_id for memory
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Initial input for the Graph
        # Note: We send the prompt as a tuple ("user", prompt) to match GraphState
        inputs = {"messages": [("user", prompt)], "retry_count": 0}
        
        final_answer = ""
        
        # 5. STREAMING: Watch the Agent think
        # 'updates' mode shows us which node is finishing its work
        for event in app.stream(inputs, config=config, stream_mode="updates"):
            for node, values in event.items():
                # Provide visual feedback on the current agent step
                status_placeholder.info(f"Agent Action: **{node.upper()}**...")
                
                # Check if the node produced a final answer
                if "answer" in values and values["answer"]:
                    # Ensure we don't catch intermediate 'no' or 'yes' grading answers
                    if node == "generate":
                        final_answer = values["answer"]

        # 6. FINAL OUTPUT: Show the answer and save it
        if final_answer:
            status_placeholder.empty() # Clear the "Thinking" status
            response_placeholder.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
        else:
            status_placeholder.error("The agent couldn't find a final answer. Check your data ingestion.")

# 7. SIDEBAR: Debug Info
with st.sidebar:
    st.subheader("Session Info")
    st.text(f"Thread ID: {st.session_state.thread_id[:8]}...")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()