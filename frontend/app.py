import sys
import os
import uuid
import re
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
if prompt := st.chat_input("Ask me about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_data = {"messages": [("user", prompt)]}

    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning..."):
            final_answer = ""
            last_event = {}
            
            # Stream the graph execution
            for event in agent_app.stream(input_data, config=config, stream_mode="values", version="v2"):
                last_event = event # Keep track for debugging
                
                # PATH 1: Direct Answer Extraction (High Priority)
                if "answer" in event and event["answer"]:
                    ans_text = str(event["answer"])
                    if len(ans_text) > 20: # Ignore short status flags like 'yes'
                        final_answer = ans_text

                # PATH 2: String Parsing Fallback (For serialized AIMessages)
                if not final_answer and "messages" in event and event["messages"]:
                    last_msg = event["messages"][-1]
                    msg_str = str(last_msg)
                    
                    if "AIMessage" in msg_str and "content=" in msg_str:
                        # Regex to pull content from AIMessage(content='...')
                        match = re.search(r"content='(.*?)'", msg_str, re.DOTALL)
                        if match:
                            final_answer = match.group(1)
                    
                    # Normal Object Path
                    elif hasattr(last_msg, "content") and getattr(last_msg, "type", "") == "ai":
                        if last_msg.content.strip():
                            final_answer = last_msg.content

            # --- Final Rendering & Cleanup ---
            if final_answer:
                # Fix escaped newlines (\n) from the serialized strings
                clean_answer = final_answer.replace('\\n', '\n').replace('\\"', '"')
                
                st.markdown(clean_answer)
                st.session_state.messages.append({"role": "assistant", "content": clean_answer})
            else:
                st.error("⚠️ Response captured in backend but failed UI render.")
                # Show raw data if something went wrong
                with st.expander("View Raw Output"):
                    st.write(last_event)