
import streamlit as st

from utils.youtube_loader import extract_video_id, get_transcript
from utils.chunker import create_chunks
from utils.embeddings import load_embedding_model
from utils.retriever import create_vector_store, retrieve_chunks
from utils.rag_chain import generate_answer
from utils.summarizer import generate_summary

# ---------------------------------------------------
# PAGE CONFIG  — wide gives room for side-by-side
# ---------------------------------------------------
st.set_page_config(
    page_title="TubeMind",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# CSS
# ---------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── BACKGROUND ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main { background: #161954 !important; }

[data-testid="stBottomBlockContainer"] {
    background: #161954 !important;
    border-top: 1px solid #121414 !important;
}
.stChatFloatingInputContainer { background: #161954 !important; }

#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── PAGE CENTERING ── */
.main .block-container {
    max-width: 1080px !important;
    padding: 2rem 2rem 5rem 2rem !important;
    margin: 0 auto !important;
}

/* ── TOP-LEVEL HORIZONTAL BLOCK = the two main cards ── */
/* Target only the direct children of the outermost stHorizontalBlock */
div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {
    background: #e6e8e1 !important;
    border: 1px solid #88B0CC !important;
    border-radius: 18px !important;
    padding: 1.6rem 1.8rem !important;
    box-shadow: 0 2px 12px rgba(60,100,140,0.13) !important;
    min-height: 0 !important;
    align-self: flex-start !important;
}

/* Prevent columns from stretching to equal height */
div[data-testid="stHorizontalBlock"]:first-of-type {
    align-items: flex-start !important;
}

/* ── HERO ── */
.hero { text-align: center; margin-bottom: 1.8rem; }
.hero-title {
    font-size: 2.3rem; font-weight: 800;
    color: #43e01f; letter-spacing: -0.025em;
    line-height: 1.15; margin-bottom: 0.45rem;
}
.hero-title .ai {
    background: linear-gradient(90deg, #E91E8C, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub { font-size: 0.88rem; color: #19e6db; margin-bottom: 1rem; }
.hero-badges { display: flex; align-items: center; justify-content: center; gap: 1.4rem; flex-wrap: wrap; }
.badge { display: flex; align-items: center; gap: 5px; font-size: 0.78rem; font-weight: 500; color: #fa0c30; }
.badge-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }

/* ── SECTION LABELS ── */
.sec-label {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.88rem; font-weight: 700;
    color: #E91E8C; margin-bottom: 0.9rem;
}
.sec-label-purple {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.88rem; font-weight: 700;
    color: #7C3AED;
}

/* ── CHAT HEADER ROW (pure HTML flex — no nested st.columns) ── */
.chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}
.chat-clear-btn {
    background: transparent;
    border: 1px solid #88B0CC;
    border-radius: 8px;
    color: #4a6a8a;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.28rem 0.8rem;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
    transition: border-color 0.18s, color 0.18s;
}
.chat-clear-btn:hover { border-color: #E91E8C; color: #E91E8C; }

/* ── URL INPUT ── */
.stTextInput label { display: none !important; }
.stTextInput > div > div {
    background: #EBF4FA !important;
    border: 1.5px solid #88B0CC !important;
    border-radius: 10px !important;
}
.stTextInput input {
    background: transparent !important; border: none !important;
    border-radius: 10px !important; color: #0f1a2e !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important;
    padding: 0.72rem 1rem !important; height: auto !important;
    width: 100% !important; box-shadow: none !important;
}
.stTextInput input::placeholder { color: #6a90b0 !important; }
.stTextInput input:focus { box-shadow: none !important; outline: none !important; }

/* ── GENERATE BUTTON ── */
.stButton > button {
    background: linear-gradient(90deg, #E91E8C 0%, #7C3AED 55%, #4F46E5 100%) !important;
    color: #FFFFFF !important; border: none !important; border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.92rem !important;
    font-weight: 700 !important; padding: 0.72rem 1.4rem !important;
    width: 100% !important; cursor: pointer !important;
    transition: opacity 0.18s, transform 0.12s !important;
    margin-top: 0.5rem !important; height: auto !important;
}
.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── DIVIDER ── */
.sec-divider { border: none; border-top: 1px solid #C0D8EC; margin: 1.1rem 0; }

/* ── SUMMARY EMPTY STATE ── */
.sum-empty { text-align: center; padding: 1.8rem 1rem; }
.sum-empty-icon { font-size: 1.8rem; display: block; margin-bottom: 0.5rem; }
.sum-empty-title { font-size: 0.86rem; font-weight: 600; color: #4a6a8a; margin-bottom: 0.2rem; }
.sum-empty-sub { font-size: 0.76rem; color: #6a90b0; }

/* ── SUMMARY TEXT ── */
.sum-content { color: #0f1a2e !important; }
.sum-content * { color: #0f1a2e !important; }
.sum-content p { color: #1e3a5a !important; font-size: 0.875rem !important; line-height: 1.75 !important; }
.sum-content h2 { color: #0f1a2e !important; font-size: 0.92rem !important; font-weight: 700 !important; }
.sum-content h3 { color: #2a4a6a !important; font-size: 0.84rem !important; font-weight: 700 !important; }
.sum-content ul, .sum-content ol { color: #2a4a6a !important; font-size: 0.85rem !important; line-height: 1.75 !important; }
.sum-content strong { color: #0f1a2e !important; font-weight: 600 !important; }

/* ── CHAT EMPTY ── */
.chat-empty { text-align: center; padding: 1.4rem 1rem; }
.chat-empty-icon { font-size: 1.3rem; display: block; margin-bottom: 0.4rem; color: #88B0CC; }
.chat-empty-title { font-size: 0.82rem; color: #4a6a8a; margin-bottom: 0.8rem; }
.chips { display: flex; flex-wrap: wrap; gap: 7px; justify-content: center; }
.chip {
    font-size: 0.73rem; color: #7C3AED;
    background: rgba(124,58,237,0.07);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 100px; padding: 4px 12px;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important;
    padding: 0 !important; margin-bottom: 0.5rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: rgba(124,58,237,0.09) !important;
    border: 1px solid rgba(124,58,237,0.18) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 0.65rem 0.9rem !important;
    max-width: 82% !important; margin-left: auto !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] p {
    color: #0f1a2e !important; font-size: 0.875rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    background: #EBF4FA !important;
    border: 1px solid #88B0CC !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 0.7rem 0.9rem !important; max-width: 88% !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] * {
    color: #0f1a2e !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] p {
    color: #1e3a5a !important; font-size: 0.875rem !important; line-height: 1.7 !important;
}
[data-testid="chatAvatarIcon-user"] { background: linear-gradient(135deg,#E91E8C,#7C3AED) !important; border-radius: 50% !important; }
[data-testid="chatAvatarIcon-assistant"] { background: linear-gradient(135deg,#7C3AED,#4F46E5) !important; border-radius: 50% !important; }

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: #EBF4FA !important; border: 1.5px solid #88B0CC !important;
    border-radius: 10px !important; padding: 0.2rem 0.2rem 0.2rem 0.6rem !important;
}
[data-testid="stChatInput"]:focus-within { border-color: #7C3AED !important; }
[data-testid="stChatInput"] textarea {
    color: #0f1a2e !important; font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important; background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #6a90b0 !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg,#7C3AED,#4F46E5) !important;
    border-radius: 8px !important; border: none !important; margin: 3px !important;
}
[data-testid="stChatInput"] button svg { color: white !important; }

/* ── ALERTS & SPINNER ── */
div[data-testid="stAlert"] { border-radius: 10px !important; font-size: 0.83rem !important; }
.stSpinner > div { color: #7C3AED !important; }

/* ── FOOTER ── */
.footer { text-align: center; margin-top: 2rem; font-size: 0.78rem; color: #2a4060; }
.footer span { color: #E91E8C; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "vector_store"    not in st.session_state: st.session_state.vector_store    = None
if "video_processed" not in st.session_state: st.session_state.video_processed = False
if "summary"         not in st.session_state: st.session_state.summary         = ""
if "messages"        not in st.session_state: st.session_state.messages        = []
if "do_clear"        not in st.session_state: st.session_state.do_clear        = False

# Handle clear flag (set via HTML button via query param workaround is unreliable;
# keep a Streamlit button hidden below instead — see right column)
if st.session_state.do_clear:
    st.session_state.messages = []
    st.session_state.do_clear = False

# ---------------------------------------------------
# HIDDEN SIDEBAR (kept for internal compatibility)
# ---------------------------------------------------
with st.sidebar:
    st.header("Video Input")
    _url = st.text_input("Enter YouTube URL")
    if st.button("Process Video"):
        if not _url:
            st.error("Please enter a YouTube URL.")
        else:
            try:
                with st.spinner("Processing…"):
                    _vid = extract_video_id(_url)
                    _tr  = get_transcript(_vid)
                    st.session_state.summary = generate_summary(_tr)
                    _chunks = create_chunks(_tr)
                    _emb    = load_embedding_model()
                    st.session_state.vector_store   = create_vector_store(_chunks, _emb)
                    st.session_state.video_processed = True
                    st.session_state.messages        = []
                st.success("Done!")
            except Exception as e:
                st.error(str(e))

# ---------------------------------------------------
# HERO
# ---------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-title">🎬 TubeMind <span class="ai">AI</span></div>
  <div class="hero-sub">Summarize any YouTube video and chat with it using AI</div>
  <div class="hero-badges">
    <span class="badge"><span class="badge-dot" style="background:#E91E8C"></span>AI Powered</span>
    <span class="badge"><span class="badge-dot" style="background:#7C3AED"></span>RAG + FAISS</span>
    <span class="badge"><span class="badge-dot" style="background:#4F46E5"></span>Groq LLM</span>
    <span class="badge"><span class="badge-dot" style="background:#0EA5E9"></span>Multi-language</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TWO-COLUMN LAYOUT  (align-items: flex-start via CSS)
# ---------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

# ════════════════════════════════════════════════════
# LEFT COLUMN — URL input + summary
# ════════════════════════════════════════════════════
with col_left:

    st.markdown('<div class="sec-label">🔗 Enter YouTube URL</div>', unsafe_allow_html=True)

    main_url = st.text_input(
        "url", placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed", key="main_url"
    )

    generate_btn = st.button("🚀 Generate Summary", key="gen_btn")

    if generate_btn:
        if not main_url:
            st.error("Please enter a YouTube URL.")
        else:
            try:
                with st.spinner("Fetching transcript…"):
                    vid_id     = extract_video_id(main_url)
                    transcript = get_transcript(vid_id)
                with st.spinner("Generating summary…"):
                    st.session_state.summary = generate_summary(transcript)
                with st.spinner("Building knowledge index…"):
                    chunks     = create_chunks(transcript)
                    embeddings = load_embedding_model()
                    st.session_state.vector_store   = create_vector_store(chunks, embeddings)
                    st.session_state.video_processed = True
                    st.session_state.messages        = []
                st.success("Video processed successfully!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">📄 Video Summary</div>', unsafe_allow_html=True)

    if st.session_state.summary:
        st.markdown('<div class="sum-content">', unsafe_allow_html=True)
        st.markdown(st.session_state.summary)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sum-empty">
          <span class="sum-empty-icon">📋</span>
          <div class="sum-empty-title">Your video summary will appear here</div>
          <div class="sum-empty-sub">Click "Generate Summary" to get started</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# RIGHT COLUMN — chat
# ════════════════════════════════════════════════════
with col_right:

    # ── Chat header: label + clear button — pure HTML flex, NO nested st.columns ──
    st.markdown("""
    <div class="chat-header">
      <div class="sec-label-purple">💬 Chat with Video</div>
    </div>""", unsafe_allow_html=True)

    # Clear button as a normal Streamlit button, styled small via CSS override
    if st.button("🗑️ Clear chat", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

    # Override ONLY the clear button to be small + ghost
    st.markdown("""
    <style>
    div[data-testid="stColumn"]:last-child button[kind="secondary"],
    [data-testid="stColumn"] + [data-testid="stColumn"] .stButton > button {
        width: auto !important;
        padding: 0.28rem 0.8rem !important;
        font-size: 0.75rem !important;
        background: transparent !important;
        border: 1px solid #88B0CC !important;
        color: #4a6a8a !important;
        margin-top: 0 !important;
        font-weight: 500 !important;
    }
    /* Target specifically the clear button by key */
    #clear_chat ~ div button,
    button[data-testid="baseButton-secondary"] {
        width: auto !important;
        padding: 0.28rem 0.85rem !important;
        font-size: 0.75rem !important;
        background: transparent !important;
        border: 1px solid #88B0CC !important;
        color: #4a6a8a !important;
        margin-top: 0 !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if not st.session_state.video_processed:
        st.markdown("""
        <div class="chat-empty">
          <span class="chat-empty-icon">💬</span>
          <div class="chat-empty-title">Ask any question about the video...</div>
          <div class="chips">
            <span class="chip">What is the main topic?</span>
            <span class="chip">Key takeaways?</span>
            <span class="chip">Explain a specific part</span>
          </div>
        </div>""", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the video...")

    if question:
        if not st.session_state.video_processed:
            st.warning("Please process a video first.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    docs   = retrieve_chunks(st.session_state.vector_store, question, k=5)
                    answer = generate_answer(question, docs)
                    st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("""
<div class="footer">
  <span>♥</span> Built with Streamlit &nbsp;•&nbsp; Powered by Groq &amp; LangChain
</div>""", unsafe_allow_html=True)


