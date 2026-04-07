"""
Mini AI PDF Assistant
Kiến trúc RAG: PDF -> Split -> Embed (HuggingFace local) -> ChromaDB -> Retrieve -> Gemini -> Answer
"""

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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
# CSS tuỳ chỉnh (Đã bỏ bong bóng chat custom cũ, dùng form chuẩn của Streamlit)
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

hr { border-color: rgba(255,255,255,0.1) !important; }

/* Làm cho input chat ghim ở đáy hòa hợp với nền màu đen */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt")
    st.divider()

    api_key = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Lấy key tại https://aistudio.google.com"
    )

    st.divider()

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

    if st.button("🔄 Xoá hội thoại", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.lc_history = []
        st.session_state.rag_chain = None
        st.rerun()

    st.markdown("""
    <div style="color:#64748b;font-size:0.75rem;text-align:center;margin-top:1rem;">
    Mini AI PDF Assistant v2.0<br>Powered by Gemini + HuggingFace
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KHỞI TẠO SESSION STATE
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "lc_history" not in st.session_state:
    st.session_state.lc_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# ─────────────────────────────────────────────
# HÀM XỬ LÝ PDF → RAG CHAIN
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_rag_chain(file_bytes: bytes, filename: str, api_key: str):
    os.environ["GOOGLE_API_KEY"] = api_key

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"pdf_{filename[:20]}"
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Bạn là trợ lý AI thông minh chuyên phân tích tài liệu PDF. "
         "Hãy trả lời câu hỏi dựa trên ngữ cảnh tài liệu được cung cấp. "
         "Nếu không tìm thấy thông tin trong tài liệu, hãy nói thẳng là không biết. "
         "Trả lời bằng tiếng Việt, súc tích và chính xác.\n\n"
         "Ngữ cảnh tài liệu:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: format_docs(retriever.invoke(x["input"]))),
            "input": RunnablePassthrough() | RunnableLambda(lambda x: x["input"]),
            "chat_history": RunnablePassthrough() | RunnableLambda(lambda x: x["chat_history"]),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📄 Mini AI PDF Assistant</h1>
    <p>Tải PDF lên → Đặt câu hỏi → AI trả lời dựa trên nội dung tài liệu</p>
</div>
""", unsafe_allow_html=True)

if not api_key:
    st.info("👈 Nhập **Google Gemini API Key** vào sidebar để bắt đầu.")
    st.stop()

if not uploaded_file:
    st.info("👈 **Tải lên file PDF** trong sidebar để bắt đầu.")
    st.stop()

# Xử lý PDF với st.status thay vì st.spinner ở toàn trang
current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
if st.session_state.processed_file != current_file_id:
    # st.status: thanh công cụ nhỏ gọn, không làm mờ UI
    with st.status("⚙️ Đang phân tích PDF... (Loading → Splitting → Embedding)", expanded=True) as status:
        try:
            rag_chain = build_rag_chain(
                file_bytes=uploaded_file.read(),
                filename=uploaded_file.name,
                api_key=api_key
            )
            st.session_state.rag_chain = rag_chain
            st.session_state.processed_file = current_file_id
            st.session_state.chat_history = []
            st.session_state.lc_history = []
            status.update(label=f"✅ Đã xử lý xong **{uploaded_file.name}**!", state="complete", expanded=False)
        except Exception as e:
            status.update(label=f"❌ Lỗi khi xử lý PDF: {e}", state="error")
            st.stop()

# ────────────────────────────────────────────────────────────
# GIAO DIỆN CHAT HIỆN ĐẠI (Tích hợp Streamlit Chat Elements)
# ────────────────────────────────────────────────────────────

# Lời chào mặc định nếu chưa có lịch sử
if not st.session_state.chat_history:
    with st.chat_message("ai", avatar="🤖"):
        st.write("Xin chào! Tôi đã đọc xong tài liệu của bạn. Hãy đặt câu hỏi về nội dung trong file nhé! 🎓")

# 1. Vẽ lại các tin nhắn cũ từ lịch sử
for role, content in st.session_state.chat_history:
    avatar = "👤" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

# 2. Ô nhập liệu chuẩn Chat Input (Ghim ở đáy màn hình)
if prompt := st.chat_input("Hỏi bất cứ điều gì về tài liệu..."):
    
    # 2a. Hiển thị tin nhắn của User ngay lập tức
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Lưu vào lịch sử User
    st.session_state.chat_history.append(("user", prompt))
    st.session_state.lc_history.append(HumanMessage(content=prompt))

    # 2b. Hiển thị Chat của AI và tiến hành luồng trả lời
    with st.chat_message("ai", avatar="🤖"):
        try:
            # Dùng st.write_stream để tạo hiệu ứng gõ từng chữ (Typewriter effect) cực kỳ mượt mà
            # thay vì spinner bắt người dùng đợi toàn bộ khung chat xuất hiện.
            stream = st.session_state.rag_chain.stream({
                "input": prompt,
                "chat_history": st.session_state.lc_history
            })
            
            # Kết quả từng tự động in ra màn hình và trả về chuỗi hoàn chỉnh
            answer = st.write_stream(stream)
            
            # Lưu lịch sử AI
            st.session_state.chat_history.append(("ai", answer))
            st.session_state.lc_history.append(AIMessage(content=answer))
        except Exception as e:
            st.error(f"❌ Lỗi khi truy vấn AI: {e}")
