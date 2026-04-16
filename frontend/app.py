import streamlit as st
import os
import sys

# Set root path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.core.agents.graph import app

st.set_page_config(page_title="Agentic RAG Pro", layout="centered")
st.title("🤖 Agentic RAG 2.0")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about LangChain..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        config = {"configurable": {"thread_id": "session_1"}}
        inputs = {"question": prompt, "retry_count": 0}
        
        # Stream the nodes as they execute
        for event in app.stream(inputs, config=config, stream_mode="updates"):
            for node, values in event.items():
                status_placeholder.info(f"Step: {node.upper()}")
                
                if "answer" in values and node == "generate":
                    full_response = values["answer"]
                    response_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})