import streamlit as st
from google import genai
from dotenv import load_dotenv
import os
import time
import random

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HCA & Co. Knowledge Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root & Background ── */
.stApp {
    background-color: #0F1117;
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 900px;
}

/* ── Header ── */
.hca-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid #2A2D3A;
    margin-bottom: 2rem;
}
.hca-badge {
    display: inline-block;
    background: linear-gradient(135deg, #C9A84C, #E8C96A);
    color: #0F1117;
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 2px;
    margin-bottom: 1rem;
}
.hca-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #F0EAD6;
    line-height: 1.2;
    margin: 0.4rem 0 0.5rem;
}
.hca-title span {
    color: #C9A84C;
}
.hca-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: #6B7280;
    font-weight: 300;
    letter-spacing: 0.03em;
}

/* ── Chat messages ── */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    margin-bottom: 1.5rem;
}

.msg-user {
    display: flex;
    justify-content: flex-end;
}
.msg-user .bubble {
    background: linear-gradient(135deg, #1E3A5F, #1A3352);
    color: #D1E8FF;
    border: 1px solid #2A5080;
    border-radius: 18px 18px 4px 18px;
    padding: 0.85rem 1.1rem;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.6;
}

.msg-bot {
    display: flex;
    justify-content: flex-start;
    gap: 0.6rem;
    align-items: flex-start;
}
.bot-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #C9A84C, #E8C96A);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.msg-bot .bubble {
    background: #1A1D27;
    color: #D4D8E2;
    border: 1px solid #2A2D3A;
    border-radius: 4px 18px 18px 18px;
    padding: 0.85rem 1.1rem;
    max-width: 80%;
    font-size: 0.92rem;
    line-height: 1.7;
}

/* ── Sources card ── */
.sources-card {
    background: #13161F;
    border: 1px solid #2A2D3A;
    border-left: 3px solid #C9A84C;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin-top: 0.6rem;
    font-size: 0.78rem;
    color: #6B7280;
    max-width: 80%;
    margin-left: 42px;
}
.sources-card strong {
    color: #C9A84C;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Input area ── */
.input-area {
    position: sticky;
    bottom: 0;
    background: #0F1117;
    padding: 1rem 0 0.5rem;
    border-top: 1px solid #1E2130;
}

/* ── Streamlit input override ── */
.stTextInput > div > div > input {
    background-color: #1A1D27 !important;
    border: 1px solid #2A2D3A !important;
    border-radius: 10px !important;
    color: #F0EAD6 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.75rem 1rem !important;
    caret-color: #C9A84C;
}
.stTextInput > div > div > input:focus {
    border-color: #C9A84C !important;
    box-shadow: 0 0 0 2px rgba(201, 168, 76, 0.15) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #4A4F63 !important;
}

/* ── Button override ── */
.stButton > button {
    background: linear-gradient(135deg, #C9A84C, #E8C96A) !important;
    color: #0F1117 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1.5rem !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #13161F !important;
    border-right: 1px solid #1E2130 !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}
.sidebar-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    color: #C9A84C;
    margin-bottom: 0.3rem;
    font-weight: 600;
}
.sidebar-section {
    background: #1A1D27;
    border: 1px solid #2A2D3A;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin-bottom: 1rem;
}
.sidebar-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 0.4rem;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.25rem 0;
}
.stat-label { font-size: 0.82rem; color: #9CA3AF; }
.stat-value { font-size: 0.82rem; color: #F0EAD6; font-weight: 500; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #3A3F52;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 1rem; }
.empty-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #4A4F63;
    margin-bottom: 0.5rem;
}
.empty-hint { font-size: 0.82rem; color: #3A3F52; }

/* ── Suggestion chips ── */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin-top: 1.2rem;
}
.chip {
    background: #1A1D27;
    border: 1px solid #2A2D3A;
    border-radius: 20px;
    padding: 0.4rem 0.9rem;
    font-size: 0.78rem;
    color: #9CA3AF;
    cursor: pointer;
    transition: border-color 0.2s;
}
.chip:hover { border-color: #C9A84C; color: #C9A84C; }

/* ── Error message ── */
.error-bubble {
    background: #1F1218;
    border: 1px solid #4A1A2A;
    border-left: 3px solid #E05070;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #F87090;
    font-size: 0.85rem;
    margin-left: 42px;
    max-width: 80%;
}

/* ── Spinner override ── */
.stSpinner > div { border-top-color: #C9A84C !important; }

/* ── Divider ── */
hr { border-color: #1E2130 !important; }

/* ── Select box override ── */
.stSelectbox > div > div {
    background-color: #1A1D27 !important;
    border-color: #2A2D3A !important;
    color: #F0EAD6 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Backend Setup ──────────────────────────────────────────────────────────────
load_dotenv()

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]

@st.cache_resource(show_spinner=False)
def load_resources():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma(persist_directory="vector_store", embedding_function=embedding_model)
    return client, db

def call_gemini_with_retry(client, prompt, model_pref="gemini-2.5-flash", retries=5):
    models_to_try = [model_pref] + [m for m in MODELS if m != model_pref]
    for model_name in models_to_try:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text, model_name
            except Exception as e:
                error_str = str(e)
                if "503" in error_str or "UNAVAILABLE" in error_str:
                    wait = (2 ** attempt) + random.uniform(1, 3)
                    time.sleep(wait)
                elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    time.sleep(20 * (attempt + 1))
                elif "404" in error_str or "NOT_FOUND" in error_str:
                    break
                else:
                    raise e
    raise Exception("All models failed. Please try again later or check your quota.")


# ── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚖️ HCA & Co.</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#4A4F63;margin-bottom:1.2rem;">Knowledge Assistant v1.0</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Model</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "", MODELS, index=0, label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Retrieval Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K chunks", min_value=1, max_value=6, value=3,
                      help="How many document chunks to retrieve per query")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Session Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row"><span class="stat-label">Queries asked</span><span class="stat-value">{st.session_state.total_queries}</span></div>
    <div class="stat-row"><span class="stat-label">Messages</span><span class="stat-value">{len(st.session_state.messages)}</span></div>
    <div class="stat-row"><span class="stat-label">Vector store</span><span class="stat-value">Chroma DB</span></div>
    <div class="stat-row"><span class="stat-label">Embeddings</span><span class="stat-value">BGE-small</span></div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    st.markdown('<div style="position:absolute;bottom:1.5rem;left:1rem;right:1rem;font-size:0.7rem;color:#3A3F52;text-align:center;">Powered by Google Gemini · LangChain · ChromaDB</div>', unsafe_allow_html=True)


# ── Main Layout ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hca-header">
    <div class="hca-badge">Internal Knowledge Base</div>
    <div class="hca-title">Hari Chand Anand <span>&</span> Co.</div>
    <div class="hca-subtitle">Ask anything about the firm — powered by your documents</div>
</div>
""", unsafe_allow_html=True)

# Load resources
with st.spinner("Loading knowledge base..."):
    try:
        client, db = load_resources()
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        st.stop()

# ── Chat History ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">⚖️</div>
        <div class="empty-text">What would you like to know?</div>
        <div class="empty-hint">Ask about the firm's history, founders, services, or policies.</div>
        <div class="chip-row">
            <div class="chip">Who founded HCA & Co.?</div>
            <div class="chip">What services do you offer?</div>
            <div class="chip">Tell me about the firm's history</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
                <div class="bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-bot">
                <div class="bot-avatar">⚖️</div>
                <div class="bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
            if msg.get("sources"):
                sources_html = " &nbsp;·&nbsp; ".join(
                    [f"<code style='background:#1E2130;padding:1px 5px;border-radius:3px;font-size:0.75rem;color:#9CA3AF;'>{s}</code>"
                     for s in msg["sources"]]
                )
                st.markdown(f"""
                <div class="sources-card">
                    <strong>Sources</strong><br/>{sources_html}
                </div>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Input ──────────────────────────────────────────────────────────────────────
def submit_question():
    """Callback: move input value to pending_question and clear the field."""
    val = st.session_state.get("question_input", "").strip()
    if val:
        st.session_state.pending_question = val
        st.session_state.question_input = ""   # clears the text box

st.markdown('<div class="input-area">', unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])

with col1:
    st.text_input(
        "",
        placeholder="Ask a question about Hari Chand Anand & Co. ...",
        label_visibility="collapsed",
        key="question_input",
        on_change=submit_question   # fires when user presses Enter
    )
with col2:
    if st.button("Ask →"):
        submit_question()

st.markdown('</div>', unsafe_allow_html=True)


# ── Query Handling ─────────────────────────────────────────────────────────────
user_q = st.session_state.pending_question

if user_q:
    # Clear pending immediately so rerun doesn't re-trigger
    st.session_state.pending_question = ""

    st.session_state.messages.append({"role": "user", "content": user_q})
    st.session_state.total_queries += 1

    with st.spinner("Searching documents and generating answer..."):
        try:
            # Retrieve chunks
            results = db.similarity_search_with_score(user_q, k=top_k)
            results = sorted(results, key=lambda x: x[1])
            docs = [doc for doc, score in results]
            context = "\n\n".join([doc.page_content for doc in docs])

            # Source labels (filename or first 40 chars)
            source_labels = []
            for doc in docs:
                src = doc.metadata.get("source", "")
                label = os.path.basename(src) if src else doc.page_content[:40] + "..."
                if label not in source_labels:
                    source_labels.append(label)

            # Prompt
            prompt = f"""
You are the official knowledge assistant of Hari Chand Anand & Co.

Answer the question using ONLY the information present in the context below.
Be clear, concise, and professional. If the answer is not in the context,
say "I don't have enough information in the knowledge base to answer that."

Context:
{context}

Question:
{user_q}

Answer:
"""
            answer, used_model = call_gemini_with_retry(client, prompt, model_pref=selected_model)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_labels,
                "model": used_model
            })

        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ {str(e)}",
                "sources": [],
                "model": None
            })

    st.rerun()