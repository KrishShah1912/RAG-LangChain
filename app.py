import streamlit as st
from src.agents.graph import app
import uuid

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Research Agent", page_icon="🤖", layout="wide")

st.title("🤖 Agentic RAG Explorer")
st.markdown("---")

# --- SESSION STATE (Memory) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    st.info(f"Thread ID: {st.session_state.thread_id}")
    st.write("This agent uses Llama 3 with self-correction logic.")

# --- CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about LangChain..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- AGENT EXECUTION ---
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        status_placeholder = st.empty()
        
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        inputs = {"question": prompt}
        
        final_answer = ""
        
        # We use app.stream to show the user what's happening
        for event in app.stream(inputs, config=config, stream_mode="updates"):
            for node, values in event.items():
                status_placeholder.write(f"⚙️ **Processing:** `{node}`...")
                if "answer" in values:
                    final_answer = values["answer"]
        
        status_placeholder.empty()
        response_placeholder.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})