"""
Mini AI PDF Assistant
Kiến trúc RAG: PDF -> Split -> Embed -> ChromaDB -> Retrieve -> Gemini -> Answer
"""

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile
import os

# ─────────────────────────────────────────────
# CẤU HÌNH TRANG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mini AI PDF Assistant",
    page_icon="📄",
    layout="wide",
)

# ─────────────────────────────────────────────
# CSS tuỳ chỉnh - giao diện tối hiện đại
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Nền tổng */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Header */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem;
}
.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.main-header p {
    color: #94a3b8;
    font-size: 1rem;
}

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0 0.5rem 15%;
    box-shadow: 0 4px 15px rgba(79,70,229,0.4);
    font-size: 0.95rem;
    line-height: 1.6;
}
.chat-ai {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: #e2e8f0;
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 15% 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
    backdrop-filter: blur(8px);
}
.chat-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.25rem;
    opacity: 0.65;
}

/* Upload zone */
.upload-info {
    background: rgba(99,102,241,0.15);
    border: 1px dashed rgba(99,102,241,0.5);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    color: #a5b4fc;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

/* Input */
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #f1f5f9 !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* Alerts */
.stSuccess { background: rgba(52,211,153,0.15) !important; }
.stWarning { background: rgba(251,191,36,0.15) !important; }
.stError   { background: rgba(248,113,113,0.15) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR - API Key + Upload PDF
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt")
    st.divider()

    # Nhập API Key
    api_key = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Lấy key tại https://aistudio.google.com"
    )

    st.divider()

    # Upload PDF
    st.markdown("### 📄 Tải lên tài liệu")
    uploaded_file = st.file_uploader(
        "Chọn file PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown(f"""
        <div class="upload-info">
        📂 <strong>{uploaded_file.name}</strong><br>
        {round(uploaded_file.size / 1024, 1)} KB
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Nút reset
    if st.button("🔄 Xoá hội thoại", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.qa_chain = None
        st.rerun()

    st.markdown("""
    <div style="color:#64748b;font-size:0.75rem;text-align:center;margin-top:1rem;">
    Mini AI PDF Assistant v1.0<br>Powered by Gemini 1.5 Flash
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KHỞI TẠO SESSION STATE
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# ─────────────────────────────────────────────
# HÀM XỬ LÝ PDF → RAG CHAIN
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_rag_chain(file_bytes: bytes, filename: str, api_key: str):
    """
    Quy trình RAG đầy đủ:
    1. Document Loading  - Đọc PDF từ bytes
    2. Text Splitting    - Chia nhỏ văn bản
    3. Vector Storage    - Tạo embedding & lưu vào ChromaDB (in-memory)
    4. Retrieval Chain   - Thiết lập chain hỏi-đáp có memory
    """
    os.environ["GOOGLE_API_KEY"] = api_key

    # ── BƯỚC 1: Document Loading ──────────────────
    # Lưu tạm file PDF ra ổ đĩa để PyPDFLoader đọc
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()          # List[Document], mỗi Document = 1 trang
    os.unlink(tmp_path)           # Xoá file tạm

    # ── BƯỚC 2: Text Splitting ────────────────────
    # RecursiveCharacterTextSplitter ưu tiên tách theo đoạn/câu trước,
    # chỉ tách thô theo ký tự nếu chunk vẫn quá lớn → giữ ngữ nghĩa tốt hơn
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Độ dài tối đa mỗi chunk (ký tự)
        chunk_overlap=200,     # Overlap để không mất ngữ cảnh giữa chunks
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # ── BƯỚC 3: Vector Storage ────────────────────
    # Tạo embedding bằng Google Generative AI Embeddings
    # Lưu vào ChromaDB chạy in-memory (không cần server riêng)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"pdf_{filename[:20]}"
    )

    # ── BƯỚC 4: Retrieval Chain ───────────────────
    # ConversationalRetrievalChain = Retriever + LLM + Memory (lưu lịch sử hội thoại)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,           # Thấp → trả lời chính xác, bám sát tài liệu
        google_api_key=api_key
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}   # Lấy 4 chunk liên quan nhất
        ),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )

    return qa_chain


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📄 Mini AI PDF Assistant</h1>
    <p>Tải PDF lên → Đặt câu hỏi → AI trả lời dựa trên nội dung tài liệu</p>
</div>
""", unsafe_allow_html=True)

# ── Kiểm tra điều kiện sẵn sàng ──
if not api_key:
    st.info("👈 Nhập **Google Gemini API Key** vào sidebar để bắt đầu.")
    st.stop()

if not uploaded_file:
    st.info("👈 **Tải lên file PDF** trong sidebar để bắt đầu.")
    st.stop()

# ── Xử lý PDF (chỉ khi file mới hoặc chain chưa build) ──
current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
if st.session_state.processed_file != current_file_id:
    with st.spinner("⚙️ Đang xử lý PDF... (Loading → Splitting → Embedding)"):
        try:
            qa_chain = build_rag_chain(
                file_bytes=uploaded_file.read(),
                filename=uploaded_file.name,
                api_key=api_key
            )
            st.session_state.qa_chain = qa_chain
            st.session_state.processed_file = current_file_id
            st.session_state.chat_history = []
            st.success(f"✅ Đã xử lý xong **{uploaded_file.name}**! Hãy đặt câu hỏi.")
        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý PDF: {e}")
            st.stop()

# ── Hiển thị hội thoại ──
st.markdown("### 💬 Hội thoại")

if not st.session_state.chat_history:
    st.markdown("""
    <div class="chat-ai">
        <div class="chat-label">🤖 AI Assistant</div>
        Xin chào! Tôi đã đọc xong tài liệu của bạn. Hãy đặt câu hỏi về nội dung trong file PDF nhé! 🎓
    </div>
    """, unsafe_allow_html=True)
else:
    for role, content in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"""
            <div class="chat-user">
                <div class="chat-label">👤 Bạn</div>
                {content}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-ai">
                <div class="chat-label">🤖 AI Assistant</div>
                {content}
            </div>
            """, unsafe_allow_html=True)

# ── Input câu hỏi ──
st.divider()
with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([5, 1])
    with cols[0]:
        user_question = st.text_input(
            "Câu hỏi",
            placeholder="Hỏi bất cứ điều gì về nội dung tài liệu...",
            label_visibility="collapsed"
        )
    with cols[1]:
        submitted = st.form_submit_button("Gửi ➤", use_container_width=True)

# ── Xử lý câu hỏi ──
if submitted and user_question.strip():
    with st.spinner("🔍 AI đang phân tích..."):
        try:
            result = st.session_state.qa_chain.invoke({"question": user_question})
            answer = result["answer"]

            # Lưu vào lịch sử
            st.session_state.chat_history.append(("user", user_question))
            st.session_state.chat_history.append(("ai", answer))
            st.rerun()

        except Exception as e:
            st.error(f"❌ Lỗi khi truy vấn AI: {e}")
