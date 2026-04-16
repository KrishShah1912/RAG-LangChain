import streamlit as st
import uuid
import sys
import os

# ---------- PATH SETUP ----------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.core.agents.graph import app

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Agentic RAG Pro+", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: white;
    }
    .chat-container {
        max-width: 900px;
        margin: auto;
    }
    .user-msg {
        background: #2563eb;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0;
        text-align: right;
    }
    .assistant-msg {
        background: #334155;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0;
    }
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("⚙️ Settings")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("Agentic RAG system with streaming + memory")

# ---------- HEADER ----------
st.markdown('<div class="title">⚡ Agentic RAG Pro+</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Retrieval + AI Reasoning + Streaming UI</div>', unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# ---------- CHAT HISTORY ----------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-msg">{msg["content"]}</div>', unsafe_allow_html=True)

# ---------- INPUT ----------
prompt = st.chat_input("Ask anything about LangChain, RAG, or AI...")

if prompt:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message instantly
    st.markdown(f'<div class="user-msg">{prompt}</div>', unsafe_allow_html=True)

    # Assistant response container
    response_placeholder = st.empty()
    status_placeholder = st.empty()

    with status_placeholder.container():
        with st.status("🤖 Thinking...", expanded=True) as status:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            final_answer = ""

            for event in app.stream({"question": prompt}, config=config, stream_mode="updates"):
                node = list(event.keys())[0]
                status.update(label=f"⚙️ {node.upper()} in progress...")

                if node == "generate":
                    final_answer = event[node].get("answer", "")

            status.update(label="✅ Done", state="complete", expanded=False)

    # Stream-like typing effect
    displayed_text = ""
    for char in final_answer:
        displayed_text += char
        response_placeholder.markdown(f'<div class="assistant-msg">{displayed_text}</div>', unsafe_allow_html=True)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": final_answer})

st.markdown('</div>', unsafe_allow_html=True)

