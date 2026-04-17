import sys
import os
import uuid
import streamlit as st

# Fix Python Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.agents.graph import app

st.set_page_config(page_title="Agentic RAG v2", layout="wide")
st.title("🚀 Enterprise Agentic RAG")

# 1. Session & Memory Initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.info(f"Session Thread: {st.session_state.thread_id}")
    if st.button("New Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# 2. Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Input Handling
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_data = {"messages": [("user", prompt)]}

    with st.spinner("Agent searching & thinking..."):
        final_answer = ""
        # The loop safely handles streaming chunks
        for event in app.stream(input_data, config=config, stream_mode="values"):
            if event and "answer" in event:
                final_answer = event["answer"]

        if final_answer:
            with st.chat_message("assistant"):
                st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})