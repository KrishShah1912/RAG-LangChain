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
if prompt := st.chat_input("Ask me about the documents or search the web..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Configuration for LangGraph Memory
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_data = {"messages": [("user", prompt)]}

    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning..."):
            final_answer = ""
            
            # Streaming the graph execution
            # version="v2" provides the standardized event structure for 2026
            for event in agent_app.stream(input_data, config=config, stream_mode="values", version="v2"):
                
                # OPTION A: Extract from your custom 'answer' key
                if "answer" in event and event["answer"]:
                    # We check length to avoid 'yes'/'no' grading flags
                    if len(str(event["answer"])) > 10:
                        final_answer = event["answer"]
                
                # OPTION B: Extract from the messages list (standard LangGraph)
                elif "messages" in event and event["messages"]:
                    last_msg = event["messages"][-1]
                    
                    # Ensure we grab content from the AI, not the user/tool
                    msg_content = getattr(last_msg, "content", None)
                    msg_type = getattr(last_msg, "type", "")
                    
                    if msg_content and msg_type == "ai":
                        final_answer = msg_content

            # --- Final Rendering ---
            if final_answer:
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            else:
                # Emergency Fallback: If no AI message found, look for any message content
                if "messages" in event and event["messages"]:
                    fallback_content = event["messages"][-1].content
                    if fallback_content:
                        st.markdown(fallback_content)
                        st.session_state.messages.append({"role": "assistant", "content": fallback_content})
                    else:
                        st.error("⚠️ Graph completed but returned empty content.")
                else:
                    st.error("⚠️ Could not retrieve answer. Check graph logs.")