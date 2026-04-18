import sys
import os
import uuid
import streamlit as st

# ==========================================
# 1. CLOUD PATH RESOLUTION
# ==========================================
# Ensures the 'backend' folder is found on Streamlit Cloud
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from backend.core.agents.graph import app as agent_app
except ImportError as e:
    st.error(f"❌ Backend Import Error: {e}")
    st.info("Ensure your 'backend' folder contains an '__init__.py' file.")
    st.stop()

# ==========================================
# 2. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="Enterprise RAG v2", layout="wide", page_icon="🚀")

# Custom CSS for a cleaner chat look
st.markdown("""
    <style>
    .stChatMessage { border-radius: 12px; margin-bottom: 10px; padding: 15px; }
    .stChatInputContainer { padding-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. SESSION STATE INITIALIZATION
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 4. SIDEBAR STATUS
# ==========================================
with st.sidebar:
    st.title("⚙️ System Control")
    st.info(f"**Session ID:**\n`{st.session_state.thread_id}`")
    
    st.divider()
    
    # Quick API Check
    for key in ["GROQ_API_KEY", "TAVILY_API_KEY", "COHERE_API_KEY"]:
        if key in st.secrets or os.getenv(key):
            st.success(f"{key} Active ✅")
        else:
            st.error(f"{key} Missing ❌")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ==========================================
# 5. CHAT INTERFACE
# ==========================================
st.title("🚀 Enterprise Agentic RAG")
st.caption("Hybrid RAG + Cohere Reranking + Tavily Web Search")

# Display historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User Input & Agent Response ---
# --- User Input & Agent Response ---
if prompt := st.chat_input("Ask me about the documents or search the web..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_data = {"messages": [("user", prompt)]}

    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning..."):
            final_answer = ""
            
            # Use a container for the streaming event data
            for event in agent_app.stream(input_data, config=config, stream_mode="values", version="v2"):
                
                # Check for the custom 'answer' key first
                # We use 'temp_answer' to avoid losing data if the loop keeps spinning
                if "answer" in event and isinstance(event["answer"], str):
                    if len(event["answer"]) > 10:
                        final_answer = event["answer"]
                
                # Check messages list as fallback
                if "messages" in event and event["messages"]:
                    last_msg = event["messages"][-1]
                    # Robust check for AI content
                    if getattr(last_msg, "type", "") == "ai" and last_msg.content:
                        if len(last_msg.content) > 10:
                            final_answer = last_msg.content

            # --- Final Rendering ---
            if final_answer:
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            else:
                # Debugging: If we got here, let's see what the LAST event actually was
                st.error("⚠️ No AI message was captured.")
                with st.expander("Debug: Final State"):
                    st.write(event)