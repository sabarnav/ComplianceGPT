import os
import json
import time
import subprocess
import streamlit as st

from modules.rag_engine import get_rag_engine

# ==========================================
# Configuration & Setup
# ==========================================

CHAT_HISTORY_DIR = "chat_history"
CHAT_HISTORY_FILE = os.path.join(CHAT_HISTORY_DIR, "history.json")
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# ==========================================
# Chat History Functions
# ==========================================

def load_chat_history():
    """Load chat history from file"""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except:
            return []
    return []

def save_chat_history(chat_history):
    """Save chat history to file"""
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(chat_history, file, indent=4)

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="ComplianceGPT",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================
# Session State
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = None

if "suggested_question" not in st.session_state:
    st.session_state.suggested_question = None

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = get_rag_engine()

# ==========================================
# Sidebar
# ==========================================

with st.sidebar:
    st.title("🛡️ ComplianceGPT")
    
    st.divider()
    
    # ------------------------
    # Navigation
    # ------------------------
    st.subheader("Navigation")
    
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.selected_chat = None
        st.session_state.suggested_question = None
        st.rerun()
    
    if st.button("💬 New Chat", use_container_width=True):
        st.session_state.selected_chat = None
        st.session_state.suggested_question = None
        st.rerun()
    
    st.divider()
    
    # ------------------------
    # Knowledge Base Status
    # ------------------------
    st.subheader("📊 Knowledge Base")
    
    try:
        doc_count = st.session_state.rag_engine.knowledge_manager.get_document_count()
        if doc_count > 0:
            st.success(f"✅ {doc_count} chunks indexed")
        else:
            st.warning("⚠️ No documents found")
    except:
        st.error("❌ Knowledge base not available")
    
    st.divider()
    
    # ------------------------
    # Upload Document
    # ------------------------
    st.subheader("📂 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["txt", "pdf"]
    )
    
    if uploaded_file is not None:
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", uploaded_file.name)
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Auto-ingest after upload
        if st.button("📚 Index Document", use_container_width=True):
            with st.spinner("Indexing document..."):
                try:
                    from modules.knowledge_manager import KnowledgeManager
                    km = KnowledgeManager()
                    chunks = km.add_document(save_path)
                    if chunks > 0:
                        st.success(f"✅ Indexed {chunks} chunks")
                        st.rerun()
                    else:
                        st.error("❌ Failed to index document")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # ------------------------
    # Available Documents
    # ------------------------
    st.subheader("📄 Available Documents")
    
    if os.path.exists("data"):
        files = os.listdir("data")
        if files:
            for file in files[:5]:  # Show up to 5
                display_name = file[:25] + "..." if len(file) > 25 else file
                st.caption(f"📄 {display_name}")
            if len(files) > 5:
                st.caption(f"... and {len(files) - 5} more")
        else:
            st.info("No documents uploaded")
    
    st.divider()
    
    # ------------------------
    # Clear Chats
    # ------------------------
    if st.button("🧹 Clear Chats", use_container_width=True):
        st.session_state.chat_history = []
        save_chat_history([])
        st.session_state.selected_chat = None
        st.session_state.suggested_question = None
        st.rerun()

# ==========================================
# Main Page
# ==========================================

# REMOVED: "with Grok" from the title
st.title("🛡️ ComplianceGPT")
st.caption("AI-Powered Cybersecurity Compliance Assistant")

# REMOVED: Model display
# st.caption(f"🤖 Model: {model}")  ← This line is removed

st.divider()

# ==========================================
# Selected Chat
# ==========================================

if st.session_state.selected_chat is not None:
    with st.chat_message("user"):
        st.markdown(st.session_state.selected_chat["question"])
    
    with st.chat_message("assistant"):
        st.markdown(st.session_state.selected_chat["answer"])
        
        # Show sources if available
        if "sources" in st.session_state.selected_chat:
            with st.expander("📚 Sources"):
                for source in st.session_state.selected_chat["sources"]:
                    st.caption(source)
    
    st.divider()

# ==========================================
# Suggested Questions
# ==========================================

if st.session_state.selected_chat is None:
    st.markdown("### 💡 Try asking")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔑 Password Policy", use_container_width=True):
            st.session_state.suggested_question = "What is the password policy?"
        
        if st.button("🔐 Access Control", use_container_width=True):
            st.session_state.suggested_question = "Explain access control."
    
    with col2:
        if st.button("🚨 Incident Response", use_container_width=True):
            st.session_state.suggested_question = "What is incident response?"
        
        if st.button("📘 ISO 27001", use_container_width=True):
            st.session_state.suggested_question = "Explain ISO 27001."

# ==========================================
# Chat Input
# ==========================================

user_question = st.chat_input("Ask a compliance question...")

# Suggested Question
if st.session_state.suggested_question is not None:
    user_question = st.session_state.suggested_question
    st.session_state.suggested_question = None

# ==========================================
# Generate AI Response
# ==========================================

if user_question:
    # Clear selected history
    st.session_state.selected_chat = None
    
    # Show User Message
    with st.chat_message("user"):
        st.markdown(user_question)
    
    # Generate Answer
    with st.spinner("🔍 Searching compliance documents..."):
        try:
            engine = st.session_state.rag_engine
            answer, sources = engine.ask(user_question)
        except Exception as e:
            answer = f"Error: {str(e)}"
            sources = []
    
    # Show Assistant Message
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        # Simulate typing
        for word in answer.split():
            full_response += word + " "
            placeholder.markdown(full_response + "▌")
            time.sleep(0.01)
        
        placeholder.markdown(full_response)
        
        # Show sources
        if sources:
            with st.expander("📚 Sources"):
                for source in sources:
                    st.caption(source)
    
    # Save Conversation
    conversation = {
        "question": user_question,
        "answer": answer,
        "sources": sources
    }
    
    st.session_state.chat_history.append(conversation)
    save_chat_history(st.session_state.chat_history)
    st.session_state.selected_chat = conversation
    
    st.rerun()

