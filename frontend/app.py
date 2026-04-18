import sys
import os
import uuid
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from backend.core.agents.graph import app as agent_app
except ImportError as e:
    st.error(f" Backend Import Error: {e}")
    st.info("Ensure your 'backend' folder contains an '__init__.py' file.")
    st.stop()

st.set_page_config(page_title="Enterprise RAG v2", layout="wide", page_icon="🚀")

st.markdown("""
    <style>
    .stChatMessage { border-radius: 12px; margin-bottom: 10px; padding: 15px; }
    .stChatInputContainer { padding-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("⚙️ System Control")
    st.info(f"**Session ID:**\n`{st.session_state.thread_id}`")
    st.divider()
    
    for key in ["GROQ_API_KEY", "TAVILY_API_KEY", "COHERE_API_KEY"]:
        if key in st.secrets or os.getenv(key):
            st.success(f"{key} Active ")
        else:
            st.error(f"{key} Missing ")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

st.title("🚀 Enterprise Agentic RAG")
st.caption("Hybrid RAG + Cohere Reranking + Tavily Web Search")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_data = {"messages": [("user", prompt)]}

    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning..."):
            final_answer = ""
            last_event_captured = {}
            
            for event in agent_app.stream(input_data, config=config, stream_mode="values", version="v2"):
                last_event_captured = event 
                
                payload = event.get("data", event) 

                if "answer" in payload and payload["answer"]:
                    val = str(payload["answer"])
                    if len(val) > 20: 
                        final_answer = val
                
                if not final_answer and "messages" in payload and payload["messages"]:
                    msgs = payload["messages"]
                    
                    try:
                        if isinstance(msgs, dict):
                            last_idx = max(msgs.keys())
                            last_msg = msgs[last_idx]
                        else:
                            last_msg = msgs[-1]
                        
                        msg_str = str(last_msg)
                        
                        if "AIMessage" in msg_str and "content='" in msg_str:
                            start_idx = msg_str.find("content='") + 9
                            end_idx = msg_str.find("', additional_kwargs=")
                            if end_idx == -1: end_idx = msg_str.rfind("'")
                            
                            extracted = msg_str[start_idx:end_idx]
                            if len(extracted) > 20:
                                final_answer = extracted
                    except Exception:
                        pass

            if final_answer:
                clean_answer = (final_answer
                                .replace('\\n', '\n')
                                .replace("\\'", "'")
                                .replace('\\"', '"'))
                
                st.markdown(clean_answer)
                st.session_state.messages.append({"role": "assistant", "content": clean_answer})
            else:
                st.error("Response found in backend but extraction logic failed.")
                with st.expander("Debug Raw Output"):
                    st.write(last_event_captured)