import streamlit as st
from pathlib import Path
from backend.ingest import extract_text_by_page
from backend.chunking import chunk_pages
from backend.vector_store_faiss import FaissVectorStore
from backend.rag_service import rag_answer
from backend.handbook_service import generate_handbook


# -----------------------------
# Streamlit UI setup
# -----------------------------
st.set_page_config(page_title="Handbook Generator", layout="wide")
st.title("AI Handbook Generator")

# -----------------------------
# Persistent storage directories
# -----------------------------
BASE_DIR = Path(__file__).parent
PDF_STORAGE_DIR = BASE_DIR / "storage" / "pdfs"
DATA_DIR = BASE_DIR / "storage" / "data"

PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Session state initialization
# -----------------------------
# Tracks PDFs available for preview/indexing in this session
if "uploaded_pdf_paths" not in st.session_state:
    st.session_state.uploaded_pdf_paths = []

# Prevent double indexing the same PDF within one session
if "indexed_pdf_paths" not in st.session_state:
    st.session_state.indexed_pdf_paths = []

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Rehydrate PDFs list after restart
# -----------------------------
# session_state dies on restart, but disk persists
if not st.session_state.uploaded_pdf_paths:
    existing_pdfs = sorted(PDF_STORAGE_DIR.glob("*.pdf"))
    st.session_state.uploaded_pdf_paths = [str(p) for p in existing_pdfs]

# -----------------------------
# Upload UI
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    new_count = 0
    for uploaded_file in uploaded_files:
        save_path = PDF_STORAGE_DIR / uploaded_file.name

        # Persist bytes to disk
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        path_str = str(save_path)
        if path_str not in st.session_state.uploaded_pdf_paths:
            st.session_state.uploaded_pdf_paths.append(path_str)
            new_count += 1

    st.success(f"Saved {new_count} new PDF(s) to disk.")

# -----------------------------
# Show persisted PDFs
# -----------------------------
st.subheader("Persisted PDFs (source of knowledge)")
if st.session_state.uploaded_pdf_paths:
    for p in st.session_state.uploaded_pdf_paths:
        st.write("•", p)
else:
    st.info("No PDFs uploaded yet.")

# -----------------------------
# Extraction / chunking preview
# -----------------------------
st.subheader("Extraction / Chunking test")

if st.session_state.uploaded_pdf_paths:
    pdf_for_preview = st.selectbox(
        "Choose a PDF for extraction/chunk preview",
        st.session_state.uploaded_pdf_paths,
        key="pdf_preview"
    )

    colA, colB = st.columns(2)

    with colA:
        if st.button("Extract text (page-by-page)"):
            pages = extract_text_by_page(pdf_for_preview)
            st.write(f"Extracted {len(pages)} pages.")

            st.write("Preview (first 2 pages):")
            for item in pages[:2]:
                st.markdown(f"**Page {item['page']}**")
                st.text(item["text"][:1500] if item["text"] else "[No text extracted]")

    with colB:
        if st.button("Extract + Chunk"):
            pages = extract_text_by_page(pdf_for_preview)
            chunks = chunk_pages(pages, source_path=pdf_for_preview)

            st.write(f"Chunks created: {len(chunks)}")
            st.write("Preview (first 2 chunks):")

            for c in chunks[:2]:
                st.markdown(f"**Page {c.page} | Chunk {c.chunk_index}**")
                st.text(c.text[:1500] if c.text else "[Empty chunk]")
else:
    st.info("Upload a PDF to test extraction/chunking.")

# -----------------------------
# FAISS Indexing + Search test
# -----------------------------
st.subheader("Index & Search (FAISS test)")

# Create/load local store (persists index + metadata on disk)
store = FaissVectorStore(store_dir=str(DATA_DIR))

if st.session_state.uploaded_pdf_paths:
    pdf_for_index = st.selectbox(
        "Choose a PDF to index",
        st.session_state.uploaded_pdf_paths,
        key="pdf_index"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Index selected PDF"):
            # Guard against duplicate indexing in the same session
            if pdf_for_index in st.session_state.indexed_pdf_paths:
                st.warning("Already indexed this PDF in this session. Reset index to re-index.")
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

                st.success(f"Indexed {len(chunk_dicts)} chunks from {Path(pdf_for_index).name}")

    with col2:
        if st.button("Reset index"):
            store.reset()
            st.session_state.indexed_pdf_paths = []
            st.warning("Index cleared (FAISS + metadata reset).")

    query = st.text_input("Search query (retrieval test)")
    if query:
        results = store.search(query, k=5)

        if not results:
            st.info("No results. Index a PDF first.")
        else:
            for r in results:
                st.markdown(
                    f"**Score:** {r['score']:.3f} | "
                    f"**Page:** {r['page']} | "
                    f"**Chunk:** {r['chunk_index']} | "
                    f"**Source:** {Path(r['source_path']).name}"
                )
                st.text(r["text"][:1200])
                st.divider()
else:
    st.info("Upload PDFs first to index and search.")

# HandBook Trigger

st.subheader("Handbook Generation (20k+)")

topic = st.text_input("Handbook topic (example: Retrieval-Augmented Generation)")

if st.button("Generate handbook"):
    with st.spinner("Generating handbook (section-by-section)..."):
        try:
            handbook_text, saved_path = generate_handbook(topic, store)
            st.success(f"Saved to: {saved_path}")
            st.text_area("Handbook output (preview)", handbook_text, height=400)
        except Exception as e:
            st.error(f"Handbook generation failed: {e}")


# -----------------------------
# Chat (RAG) — thin UI handler
# -----------------------------
st.subheader("Chat (RAG)")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask a question grounded in your PDFs")

if user_input:
    # Store + show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Call reusable RAG service
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

    # Store assistant reply
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Debug: show exactly what evidence was used
    with st.expander("Retrieved context (debug)"):
        st.text(context_text if context_text else "[No retrieved context]")
