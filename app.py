import streamlit as st
from pathlib import Path
from datetime import datetime

from backend.ingest import extract_text_by_page
from backend.chunking import chunk_pages
from backend.vector_store_faiss import FaissVectorStore
from backend.rag_service import rag_answer
from backend.handbook_service import generate_handbook

# -----------------------------
# Page config + style
# -----------------------------
st.set_page_config(page_title="Handbook Generator", page_icon="📘", layout="wide")

st.markdown(
    """
    <style>
      .small {opacity: 0.8; font-size: 0.9rem;}
      .muted {opacity: 0.7;}
      .pill {display:inline-block; padding:0.15rem 0.55rem; border-radius:999px;
             border:1px solid rgba(255,255,255,0.15); font-size:0.85rem;}
      .ok {background: rgba(0, 200, 0, 0.12);}
      .warn {background: rgba(255, 200, 0, 0.12);}
      .bad {background: rgba(255, 0, 0, 0.12);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📘 AI Handbook Generator")
st.caption("Upload PDFs → Index → Chat with citations → Generate a long handbook")

# -----------------------------
# Persistent storage directories
# -----------------------------
BASE_DIR = Path(__file__).parent
PDF_STORAGE_DIR = BASE_DIR / "storage" / "pdfs"
DATA_DIR = BASE_DIR / "storage" / "data"
HANDBOOK_DIR = BASE_DIR / "storage" / "handbooks"

PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
HANDBOOK_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Session state
# -----------------------------
if "uploaded_pdf_paths" not in st.session_state:
    st.session_state.uploaded_pdf_paths = []
if "indexed_pdf_paths" not in st.session_state:
    st.session_state.indexed_pdf_paths = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_handbook_path" not in st.session_state:
    st.session_state.last_handbook_path = ""

# Rehydrate PDF list from disk on restart
if not st.session_state.uploaded_pdf_paths:
    st.session_state.uploaded_pdf_paths = [str(p) for p in sorted(PDF_STORAGE_DIR.glob("*.pdf"))]

# Vector store (local)
store = FaissVectorStore(store_dir=str(DATA_DIR))

# -----------------------------
# Sidebar: Upload + Index
# -----------------------------
with st.sidebar:
    st.header("Setup")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        new_count = 0
        for f in uploaded_files:
            save_path = PDF_STORAGE_DIR / f.name
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())

            p = str(save_path)
            if p not in st.session_state.uploaded_pdf_paths:
                st.session_state.uploaded_pdf_paths.append(p)
                new_count += 1

        st.success(f"Saved {new_count} new PDF(s).")

    st.divider()

    # PDF selection for indexing
    if st.session_state.uploaded_pdf_paths:
        pdf_for_index = st.selectbox(
            "Select PDF to index",
            st.session_state.uploaded_pdf_paths,
            format_func=lambda x: Path(x).name,
            key="pdf_index_sidebar",
        )

        indexed = pdf_for_index in st.session_state.indexed_pdf_paths
        badge = "Indexed ✅" if indexed else "Not indexed ⚠️"
        css = "pill ok" if indexed else "pill warn"
        st.markdown(f"<span class='{css}'>{badge}</span>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Index", use_container_width=True):
                if indexed:
                    st.warning("Already indexed in this session. Reset to re-index.")
                else:
                    pages = extract_text_by_page(pdf_for_index)
                    chunks = chunk_pages(pages, source_path=pdf_for_index)

                    chunk_dicts = [{
                        "text": c.text,
                        "page": c.page,
                        "chunk_index": c.chunk_index,
                        "source_path": c.source_path
                    } for c in chunks]

                    store.add_chunks(chunk_dicts)
                    st.session_state.indexed_pdf_paths.append(pdf_for_index)
                    st.success(f"Indexed {len(chunk_dicts)} chunks.")

        with col_b:
            if st.button("Reset index", use_container_width=True):
                store.reset()
                st.session_state.indexed_pdf_paths = []
                st.warning("Index cleared.")

        st.caption("Tip: Index at least 1 PDF before chatting or generating a handbook.")
    else:
        st.info("Upload PDFs to enable indexing.")

    st.divider()
    st.subheader("Files")

    if st.session_state.uploaded_pdf_paths:
        for p in st.session_state.uploaded_pdf_paths[:8]:
            st.write("•", Path(p).name)
        if len(st.session_state.uploaded_pdf_paths) > 8:
            st.write(f"… +{len(st.session_state.uploaded_pdf_paths) - 8} more")
    else:
        st.caption("No PDFs yet.")

# -----------------------------
# Main area: Tabs
# -----------------------------
tab_chat, tab_handbook, tab_debug = st.tabs(["💬 Chat", "📄 Handbook", "🧪 Debug"])

# -----------------------------
# Chat tab (thin UI)
# -----------------------------
with tab_chat:
    if not st.session_state.indexed_pdf_paths:
        st.warning("Index at least one PDF first (use the sidebar).")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask a question grounded in your PDFs (citations required)")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            try:
                answer, retrieved, context_text = rag_answer(
                    question=user_input,
                    store=store,
                    k=6
                )
            except Exception as e:
                answer, retrieved, context_text = f"Error during RAG: {e}", [], ""

            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Keep debug hidden by default
        with st.expander("Show retrieved evidence (debug)"):
            st.text(context_text if context_text else "[No retrieved context]")

# -----------------------------
# Handbook tab
# -----------------------------
with tab_handbook:
    st.subheader("Generate a long handbook")

    topic = st.text_input(
        "Topic",
        value="Retrieval-Augmented Generation",
        help="Keep it specific to get better structure."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        generate = st.button("Generate handbook", type="primary", use_container_width=True)

    with col2:
        if st.session_state.last_handbook_path:
            st.markdown(f"<span class='pill ok'>Last output saved ✅</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='pill warn'>No output yet</span>", unsafe_allow_html=True)

    if generate:
        if not st.session_state.indexed_pdf_paths:
            st.error("Index at least one PDF first (sidebar).")
        else:
            with st.spinner("Generating section-by-section… (this can take a while)"):
                try:
                    handbook_text, saved_path = generate_handbook(topic, store)
                    st.session_state.last_handbook_path = saved_path
                    st.success(f"Saved: {saved_path}")
                    st.text_area("Preview", handbook_text[:12000], height=400)
                    st.caption("Preview shows the first part only. Open the saved .md file for full output.")
                except Exception as e:
                    st.error(f"Handbook generation failed: {e}")

    # Show latest handbook file (if exists)
    latest_files = sorted(HANDBOOK_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if latest_files:
        latest = latest_files[0]
        st.caption(f"Latest handbook: {latest.name} • modified {datetime.fromtimestamp(latest.stat().st_mtime)}")
    else:
        st.caption("No handbook files found yet.")

# -----------------------------
# Debug tab (only for you)
# -----------------------------
with tab_debug:
    st.subheader("Diagnostics")

    st.write("Indexed PDFs this session:", len(st.session_state.indexed_pdf_paths))
    st.write("PDFs available:", len(st.session_state.uploaded_pdf_paths))

    st.divider()
    st.write("Storage paths:")
    st.code(
        f"PDFs: {PDF_STORAGE_DIR}\n"
        f"Index: {DATA_DIR}\n"
        f"Handbooks: {HANDBOOK_DIR}",
        language="text"
    )

    st.divider()
    if st.session_state.uploaded_pdf_paths:
        pdf_for_preview = st.selectbox(
            "Preview extraction/chunking for a PDF",
            st.session_state.uploaded_pdf_paths,
            format_func=lambda x: Path(x).name,
            key="pdf_preview_debug",
        )

        colA, colB = st.columns(2)
        with colA:
            if st.button("Extract preview"):
                pages = extract_text_by_page(pdf_for_preview)
                st.write(f"Extracted pages: {len(pages)}")
                for item in pages[:2]:
                    st.markdown(f"**Page {item['page']}**")
                    st.text(item["text"][:1500] if item["text"] else "[No text extracted]")

        with colB:
            if st.button("Chunk preview"):
                pages = extract_text_by_page(pdf_for_preview)
                chunks = chunk_pages(pages, source_path=pdf_for_preview)
                st.write(f"Chunks: {len(chunks)}")
                for c in chunks[:2]:
                    st.markdown(f"**Page {c.page} | Chunk {c.chunk_index}**")
                    st.text(c.text[:1500] if c.text else "[Empty chunk]")
    else:
        st.info("Upload PDFs to enable debug previews.")
