import sys
import os
import uuid
import streamlit as st

# ==========================================
# 1. CLOUD PATH & IMPORT FIX
# ==========================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from backend.core.agents.graph import app
except ImportError as e:
    st.error(f"Backend Import Error: {e}")
    st.stop()

# ==========================================
# 2. SESSION & MEMORY INITIALIZATION
# ==========================================
st.set_page_config(page_title="Agentic RAG v2", layout="wide")
st.title("🚀 Enterprise Agentic RAG")

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

# ==========================================
# 3. DISPLAY HISTORY
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 4. INPUT HANDLING & STREAMING
# ==========================================
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to UI and State
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_data = {"messages": [("user", prompt)]}

    with st.chat_message("assistant"):
        with st.spinner("Agent searching & thinking..."):
            final_answer = ""
            
            # version="v2" is required for the latest LangGraph event structure
        for event in app.stream(input_data, config=config, stream_mode="values", version="v2"):
            if "messages" in event and event["messages"]:
                # 1. Get the very last message in the list
                last_msg = event["messages"][-1]
                
                # 2. Check if it's an AI Message (the actual answer)
                # Note: We check .content because some messages might be Tool calls
                if hasattr(last_msg, "content") and last_msg.content.strip():
                    # We only want to capture it if it's an AIMessage
                    # In some versions, it might be an 'ai' type or an AIMessage class
                    if getattr(last_msg, "type", "") == "ai" or "AIMessage" in str(type(last_msg)):
                        final_answer = last_msg.content

            if final_answer:
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            else:
               st.error("⚠️ The graph completed but no AI message was found.")
               st.info("Check your 'generate' node in graph.py to ensure it returns: {'messages': [AIMessage(content=...)]}")