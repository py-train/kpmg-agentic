"""
=============================================================================
KPMG Agentic AI Fundamentals Training
=============================================================================
RAG Demo  |  Multi-Turn PDF Q&A
-----------------------------------------------------------------------------
Stack:
  • LangChain v1+      – init_chat_model, init_embeddings, LCEL chains
  • ChromaDB           – local persistent vector store (langchain-chroma)
  • OpenAI             – GPT-4o-mini (chat) + text-embedding-3-small (embed)
  • PyPDF              – PDF text extraction (PyPDFLoader)
  • Gradio 4+          – interactive front-end with drag-and-drop upload

Install:
  pip install langchain langchain-core langchain-openai langchain-chroma \
              langchain-community langchain-text-splitters \
              chromadb openai pypdf gradio python-dotenv

Run:
  python rag_demo.py
  → Browser opens at http://localhost:7860
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0.  Standard library
# ---------------------------------------------------------------------------
import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# 1.  Environment  (.env file or system env)
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# 2.  LangChain v1+ imports
# ---------------------------------------------------------------------------
# --- Model initialisation (unified v1 API) ---
from langchain.chat_models  import init_chat_model
from langchain.embeddings   import init_embeddings

# --- Core message types ---
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- Prompt building ---
from langchain_core.prompts       import ChatPromptTemplate, MessagesPlaceholder

# --- Output parsing ---
from langchain_core.output_parsers import StrOutputParser

# --- Vector store ---
from langchain_chroma import Chroma

# --- Document loading & splitting ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters              import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# 3.  Gradio
# ---------------------------------------------------------------------------
import gradio as gr

# ---------------------------------------------------------------------------
# 4.  Configuration constants
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = "./kpmg_rag_chroma_db"
COLLECTION_NAME    = "kpmg_rag_collection"

CHUNK_SIZE         = 1000      # characters per chunk
CHUNK_OVERLAP      = 150       # overlap between consecutive chunks
TOP_K              = 5         # number of chunks to retrieve
FETCH_K            = 12        # MMR candidate pool size

CHAT_MODEL         = "gpt-4o-mini"
EMBED_MODEL        = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# 5.  Prompt templates
# ---------------------------------------------------------------------------

# Rewrites the latest user question into a standalone question
# that can be understood without prior chat history.
CONTEXTUALISE_SYSTEM = (
    "You are a helpful assistant. Given the conversation history below and "
    "the user's latest question, rewrite the question so it is fully "
    "self-contained (i.e. anyone could understand it without reading the "
    "history). Return ONLY the rewritten question with no explanation."
)

# Main RAG system prompt — {context} is filled at runtime
RAG_SYSTEM = (
    "You are a knowledgeable assistant helping users understand a document. "
    "Use ONLY the retrieved document excerpts below to answer the question. "
    "If the answer is not present in the excerpts, say: "
    "\"I don't have enough information in the document to answer that.\"\n\n"
    "Always be concise, accurate, and cite the page number when possible.\n\n"
    "===== Retrieved Excerpts =====\n"
    "{context}"
    "\n============================="
)

# ---------------------------------------------------------------------------
# 6.  Helper utilities
# ---------------------------------------------------------------------------

def _set_api_key(api_key: str) -> None:
    """Push the user-supplied API key into the environment."""
    key = api_key.strip()
    if key:
        os.environ["OPENAI_API_KEY"] = key


def _format_docs(docs: list) -> str:
    """
    Convert a list of LangChain Document objects into a single string
    that the LLM can reason over.  Each chunk is prefixed with its page.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "?")
        # PDF pages are 0-indexed in PyPDFLoader; display as 1-indexed
        if isinstance(page, int):
            page += 1
        parts.append(f"[Excerpt {i} – Page {page}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def _history_to_lc(history: list[dict]) -> list:
    """
    Convert Gradio's messages list (list of {"role":…,"content":…} dicts)
    into a list of LangChain message objects.
    """
    mapping = {"user": HumanMessage, "assistant": AIMessage}
    return [
        mapping[turn["role"]](content=turn["content"])
        for turn in history
        if turn["role"] in mapping
    ]

# ---------------------------------------------------------------------------
# 7.  Ingestion pipeline  –  PDF → ChromaDB
# ---------------------------------------------------------------------------

def build_vector_store(pdf_path: str, api_key: str) -> tuple[Chroma, str]:
    """
    Full ingestion pipeline:
      PyPDFLoader → RecursiveCharacterTextSplitter
        → OpenAI Embeddings → ChromaDB (local persist)

    Parameters
    ----------
    pdf_path : local filesystem path to the uploaded PDF
    api_key  : OpenAI API key (may come from UI)

    Returns
    -------
    (vector_store, human_readable_status)
    """
    _set_api_key(api_key)

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise ValueError(
            "No OpenAI API key found.  Enter it in the API Key field."
        )

    # -- Step 1: load PDF pages --
    loader   = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    # -- Step 2: split into chunks --
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    # -- Step 3: wipe old DB so re-uploads start clean --
    db_path = Path(CHROMA_PERSIST_DIR)
    if db_path.exists():
        shutil.rmtree(db_path)

    # -- Step 4: embed & store --
    # init_embeddings uses the v1 unified embeddings API
    embeddings = init_embeddings(f"openai:{EMBED_MODEL}")

    vector_store = Chroma.from_documents(
        documents         = chunks,
        embedding         = embeddings,
        collection_name   = COLLECTION_NAME,
        persist_directory = CHROMA_PERSIST_DIR,
    )

    filename = Path(pdf_path).name
    status   = (
        f"✅  Document indexed successfully!\n"
        f"    File     : {filename}\n"
        f"    Pages    : {len(raw_docs)}\n"
        f"    Chunks   : {len(chunks)}\n"
        f"    Embed    : {EMBED_MODEL}\n"
        f"    Vector DB: ChromaDB  →  {CHROMA_PERSIST_DIR}"
    )
    return vector_store, status

# ---------------------------------------------------------------------------
# 8.  RAG query pipeline  –  question + history → answer
# ---------------------------------------------------------------------------

def answer_question(
    question:     str,
    history:      list[dict],
    vector_store: Chroma,
    api_key:      str,
) -> tuple[str, list[dict]]:
    """
    History-aware RAG query pipeline:

      1. If prior history exists, rewrite the question into a standalone
         version so retrieval is not confused by pronouns / references.
      2. Retrieve TOP_K document chunks with MMR.
      3. Build RAG answer using retrieved context + full chat history.

    Parameters
    ----------
    question     : current user input string
    history      : Gradio messages list (role/content dicts)
    vector_store : loaded ChromaDB Chroma object
    api_key      : OpenAI API key

    Returns
    -------
    (answer_string, updated_history_list)
    """
    _set_api_key(api_key)

    lc_history = _history_to_lc(history)

    # -- Initialise LLM (v1 unified API) --
    llm = init_chat_model(
        CHAT_MODEL,
        model_provider = "openai",
        temperature    = 0.2,
    )

    # ---- Stage A: contextualise question --------------------------------
    # Rephrase only when there is prior history to integrate.
    if lc_history:
        ctx_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALISE_SYSTEM),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        ctx_chain          = ctx_prompt | llm | StrOutputParser()
        standalone_question = ctx_chain.invoke({
            "chat_history": lc_history,
            "question":     question,
        })
    else:
        standalone_question = question

    # ---- Stage B: retrieve relevant chunks --------------------------------
    retriever = vector_store.as_retriever(
        search_type   = "mmr",
        search_kwargs = {"k": TOP_K, "fetch_k": FETCH_K},
    )
    docs    = retriever.invoke(standalone_question)
    context = _format_docs(docs)

    # ---- Stage C: generate grounded answer --------------------------------
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM.format(context=context)),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    rag_chain = rag_prompt | llm | StrOutputParser()

    answer = rag_chain.invoke({
        "chat_history": lc_history,
        "question":     question,
    })

    # -- Append this turn to history --
    updated_history = history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]
    return answer, updated_history

# ---------------------------------------------------------------------------
# 9.  Gradio event handlers
# ---------------------------------------------------------------------------

def evt_upload(pdf_file, api_key: str, session: dict):
    """
    Triggered when a PDF is uploaded or dropped.
    Builds the vector store and resets the chat.
    """
    if pdf_file is None:
        return "⚠️  No file received.  Please upload a PDF.", session, []
    if not api_key.strip():
        return (
            "⚠️  Please enter your OpenAI API key before uploading.",
            session, []
        )
    try:
        vs, status = build_vector_store(pdf_file.name, api_key)
        session["vector_store"] = vs
        session["doc_name"]     = Path(pdf_file.name).name
        return status, session, []          # [] resets the chatbot
    except Exception as exc:
        return f"❌  Indexing failed:\n{exc}", session, []


def evt_chat(message: str, history: list, api_key: str, session: dict):
    """
    Triggered when the user sends a message.
    Returns (cleared_input, updated_chat_history).
    """
    if not message.strip():
        return "", history

    if not session.get("vector_store"):
        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content":
             "⚠️  No document loaded yet.  Please upload and index a PDF first."},
        ]
        return "", history

    if not api_key.strip():
        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content":
             "⚠️  Please enter your OpenAI API key in the left panel."},
        ]
        return "", history

    try:
        _, updated = answer_question(
            question     = message,
            history      = history,
            vector_store = session["vector_store"],
            api_key      = api_key,
        )
        return "", updated
    except Exception as exc:
        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": f"❌  Error: {exc}"},
        ]
        return "", history


def evt_clear_chat():
    """Reset the chat history only (keep the indexed document)."""
    return []


def evt_reset_all(session: dict):
    """Wipe the vector store from disk and reset everything."""
    session.clear()
    if Path(CHROMA_PERSIST_DIR).exists():
        shutil.rmtree(CHROMA_PERSIST_DIR)
    return "🗑️  Session reset.  Upload a new PDF to begin.", session, []


def evt_example_click(example_index: int):
    """Fill the message box with a pre-written example question."""
    examples = [
        "Summarise this document in 5 bullet points.",
        "What are the main topics or themes covered?",
        "What key findings, conclusions, or recommendations are made?",
        "Can you elaborate on your previous answer with more detail?",
        "What data, evidence, or examples are provided to support the claims?",
        "Are there any risks, limitations, or caveats mentioned?",
    ]
    return examples[example_index]

# ---------------------------------------------------------------------------
# 10.  Custom CSS  (KPMG dark-blue theme)
# ---------------------------------------------------------------------------
CSS = """
/* ── Page & container ── */
body, .gradio-container {
    background-color: #0b1120 !important;
    font-family: Arial, sans-serif !important;
    color: #f1f5f9 !important;
}

/* ── Top banner ── */
#kpmg-banner {
    background: linear-gradient(135deg, #020617 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-bottom: 3px solid #38bdf8;
    border-radius: 10px;
    padding: 20px 28px 16px;
    margin-bottom: 18px;
}
#kpmg-banner h1 {
    color: #38bdf8;
    font-size: 1.65rem;
    margin: 0 0 6px;
    letter-spacing: -0.02em;
}
#kpmg-banner p { color: #94a3b8; font-size: 0.82rem; margin: 0; }

/* ── Section headings inside Markdown ── */
.gr-prose h3 { color: #38bdf8 !important; font-size: 0.95rem !important; }

/* ── Labels ── */
label .svelte-1gfkn6j, .label-wrap span {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── Text inputs & password boxes ── */
input, textarea {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    color: #f1f5f9 !important;
    border-radius: 6px !important;
}
input:focus, textarea:focus {
    border-color: #38bdf8 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.2) !important;
}

/* ── Buttons ── */
button.primary { background: #38bdf8 !important; color: #0f172a !important;
                 font-weight: 700 !important; border-radius: 7px !important; }
button.primary:hover { background: #7dd3fc !important; }
button.secondary { background: #1e293b !important; color: #cbd5e1 !important;
                   border: 1px solid #334155 !important; border-radius: 7px !important; }
button.stop { background: #450a0a !important; color: #fca5a5 !important;
              border-radius: 7px !important; }

/* ── Chatbot bubbles ── */
.message.user    { background: #1e3a5f !important; color: #f1f5f9 !important;
                   border-radius: 10px !important; }
.message.bot     { background: #1e293b !important; color: #f1f5f9 !important;
                   border-radius: 10px !important;
                   border-left: 3px solid #38bdf8 !important; }

/* ── Status / log area ── */
#status-out textarea {
    background: #020617 !important;
    color: #4ade80 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.75rem !important;
    line-height: 1.6 !important;
}

/* ── Upload dropzone ── */
.upload-container, .file-preview {
    background: #0f172a !important;
    border: 2px dashed #334155 !important;
    border-radius: 8px !important;
}
.upload-container:hover { border-color: #38bdf8 !important; }

/* ── Accordion ── */
.accordion { background: #1e293b !important; border: 1px solid #334155 !important;
             border-radius: 8px !important; }

/* ── Info / metric panels ── */
.info-panel {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.8;
}
.info-panel b { color: #38bdf8; }
"""

# ---------------------------------------------------------------------------
# 11.  Static HTML blocks used in the UI
# ---------------------------------------------------------------------------
BANNER_HTML = """
<div id="kpmg-banner">
  <h1>🤖 RAG Demo &mdash; Multi-Turn PDF Q&amp;A</h1>
  <p>
    KPMG Agentic AI Fundamentals Training &nbsp;|&nbsp;
    LangChain v1+ &nbsp;&middot;&nbsp; ChromaDB &nbsp;&middot;&nbsp;
    OpenAI GPT-4o-mini &nbsp;&middot;&nbsp; Gradio
  </p>
</div>
"""

PIPELINE_HTML = """
<div class="info-panel">
  <b>⚙ Ingestion Pipeline</b><br/>
  PDF&nbsp;Upload&nbsp;→&nbsp;PyPDFLoader&nbsp;→&nbsp;RecursiveCharacterTextSplitter<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;→&nbsp;OpenAI&nbsp;Embeddings&nbsp;(text-embedding-3-small)<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;→&nbsp;ChromaDB&nbsp;Vector&nbsp;Store&nbsp;(local&nbsp;persist)<br/><br/>
  <b>🔍 Query Pipeline (per turn)</b><br/>
  User&nbsp;Question&nbsp;+&nbsp;History&nbsp;→&nbsp;Contextualiser&nbsp;(GPT-4o-mini)<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;→&nbsp;Standalone&nbsp;Question&nbsp;→&nbsp;MMR&nbsp;Retrieval&nbsp;(k=5)<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;→&nbsp;RAG&nbsp;Answer&nbsp;Chain&nbsp;→&nbsp;GPT-4o-mini&nbsp;→&nbsp;Response
</div>
"""

TIPS_HTML = """
<div class="info-panel">
  <b>💡 Tips</b><br/>
  • Upload any PDF (reports, papers, contracts, slides)<br/>
  • Ask follow-ups — the bot remembers the conversation<br/>
  • <i>\"Summarise this document in 5 bullet points\"</i><br/>
  • <i>\"What does it say about [topic]?\"</i><br/>
  • <i>\"Can you elaborate on your last answer?\"</i><br/>
  • Click <b>Clear Chat</b> to restart without re-indexing<br/>
  • Click <b>Reset All</b> to upload a different PDF
</div>
"""

# ---------------------------------------------------------------------------
# 12.  Gradio UI definition
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    ["Summarise this document in 5 bullet points."],
    ["What are the main topics or themes covered?"],
    ["What key findings, conclusions, or recommendations are made?"],
    ["Can you elaborate on your previous answer with more detail?"],
    ["What data, evidence, or examples are provided?"],
    ["Are there any risks, limitations, or caveats mentioned?"],
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="KPMG RAG Demo") as demo:

        # ── Per-session state (vector store lives here) ──────────────────
        session = gr.State({})

        # ── Top banner ───────────────────────────────────────────────────
        gr.HTML(BANNER_HTML)

        # ── Main two-column layout ────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ============================================================
            # LEFT COLUMN  –  Configuration & Upload
            # ============================================================
            with gr.Column(scale=1, min_width=300):

                # ---- API Key -------------------------------------------
                gr.Markdown("### 🔑 API Key")
                api_key = gr.Textbox(
                    label       = "OpenAI API Key",
                    placeholder = "sk-…  (kept local, never stored)",
                    type        = "password",
                    value       = os.environ.get("OPENAI_API_KEY", ""),
                    info        = "Required for embeddings and chat completions.",
                )

                # ---- Upload --------------------------------------------
                gr.Markdown("### 📄 Document")
                pdf_file = gr.File(
                    label        = "Upload PDF  (drag & drop supported)",
                    file_types   = [".pdf"],
                    file_count   = "single",
                    elem_classes = ["upload-container"],
                )
                index_btn = gr.Button(
                    "⚡  Index Document",
                    variant = "primary",
                )

                status_out = gr.Textbox(
                    label       = "Indexing Log",
                    value       = "Awaiting PDF upload…",
                    interactive = False,
                    lines       = 6,
                    max_lines   = 10,
                    elem_id     = "status-out",
                )

                # ---- Pipeline info & tips ------------------------------
                gr.HTML(PIPELINE_HTML)
                gr.HTML(TIPS_HTML)

            # ============================================================
            # RIGHT COLUMN  –  Chat interface
            # ============================================================
            with gr.Column(scale=2):

                gr.Markdown("### 💬 Conversation")

                chatbot = gr.Chatbot(
                    label            = "RAG Chat",
                    height           = 460,
                    show_label       = False,
                    # bubble_full_width= False,
                    # type             = "messages",
                    avatar_images    = (None, None),
                    # show_copy_button = True,
                    placeholder      = (
                        "<div style='text-align:center; color:#475569; padding:60px 20px'>"
                        "<div style='font-size:2.5rem'>📄</div>"
                        "<div style='margin-top:12px; font-size:1rem;'>"
                        "Upload and index a PDF, then ask questions below."
                        "</div></div>"
                    ),
                )

                # ---- Message input row ---------------------------------
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder = "Ask anything about the document…  (Enter to send)",
                        show_label  = False,
                        scale       = 5,
                        lines       = 2,
                        max_lines   = 5,
                        autofocus   = True,
                    )
                    send_btn = gr.Button(
                        "Send ➤",
                        scale   = 1,
                        variant = "primary",
                        min_width = 90,
                    )

                # ---- Action buttons ------------------------------------
                with gr.Row():
                    clear_btn = gr.Button(
                        "🗑️  Clear Chat",
                        scale   = 1,
                        variant = "secondary",
                    )
                    reset_btn = gr.Button(
                        "🔄  Reset All",
                        scale   = 1,
                        variant = "stop",
                    )

                # ---- Example questions accordion -----------------------
                with gr.Accordion("📋 Example Questions  (click to fill)", open=False):
                    gr.Markdown(
                        "<small style='color:#94a3b8'>"
                        "Click any question to fill the message box, then press Send."
                        "</small>"
                    )
                    example_box = gr.Dataset(
                        label      = "",
                        components = [msg_box],
                        samples    = EXAMPLE_QUESTIONS,
                        type       = "index",
                    )

        # ── Event wiring ─────────────────────────────────────────────────

        # Auto-index when file is dropped onto the upload widget
        pdf_file.upload(
            fn      = evt_upload,
            inputs  = [pdf_file, api_key, session],
            outputs = [status_out, session, chatbot],
        )

        # Manual index button
        index_btn.click(
            fn      = evt_upload,
            inputs  = [pdf_file, api_key, session],
            outputs = [status_out, session, chatbot],
        )

        # Send via button click
        send_btn.click(
            fn      = evt_chat,
            inputs  = [msg_box, chatbot, api_key, session],
            outputs = [msg_box, chatbot],
        )

        # Send via Enter key (submit on the Textbox)
        msg_box.submit(
            fn      = evt_chat,
            inputs  = [msg_box, chatbot, api_key, session],
            outputs = [msg_box, chatbot],
        )

        # Clear chat history (keep indexed document)
        clear_btn.click(
            fn      = evt_clear_chat,
            outputs = [chatbot],
        )

        # Full reset (clears vector DB too)
        reset_btn.click(
            fn      = evt_reset_all,
            inputs  = [session],
            outputs = [status_out, session, chatbot],
        )

        # Example question click → fill message box
        example_box.click(
            fn      = evt_example_click,
            inputs  = [example_box],
            outputs = [msg_box],
        )

    return demo


# ---------------------------------------------------------------------------
# 13.  Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  KPMG Agentic AI Fundamentals Training")
    print("  RAG Demo  –  Multi-Turn PDF Q&A")
    print("=" * 60)
    print(f"  Chat model : {CHAT_MODEL}")
    print(f"  Embed model: {EMBED_MODEL}")
    print(f"  Chunk size : {CHUNK_SIZE}  |  Overlap: {CHUNK_OVERLAP}")
    print(f"  Retrieval  : MMR  k={TOP_K}  fetch_k={FETCH_K}")
    print(f"  Vector DB  : ChromaDB  →  {CHROMA_PERSIST_DIR}")
    print("=" * 60)
    print("  Navigate to http://localhost:7860")
    print("=" * 60)

    app = build_ui()
    app.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,   # flip to True for a public Gradio link
        inbrowser   = True,
    )
