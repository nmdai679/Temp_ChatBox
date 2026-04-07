# 📄 Mini AI PDF Assistant

> Hỏi đáp thông minh với tài liệu PDF sử dụng kiến trúc RAG và Google Gemini 1.5 Flash.

---

## 🌟 Giới thiệu

**Mini AI PDF Assistant** là ứng dụng demo minh họa kiến trúc **RAG (Retrieval-Augmented Generation)**. Sinh viên chỉ cần tải file PDF tài liệu học tập lên, sau đó đặt câu hỏi bằng ngôn ngữ tự nhiên — AI sẽ trả lời chính xác dựa trên nội dung tài liệu đó, không hallucinate.

### Luồng xử lý RAG

```
PDF Upload
    │
    ▼
[1] Document Loading     ← PyPDF đọc từng trang
    │
    ▼
[2] Text Splitting       ← RecursiveCharacterTextSplitter (chunk=1000, overlap=200)
    │
    ▼
[3] Embedding + Storage  ← Google Embedding-001 → ChromaDB (in-memory)
    │
    ▼
[4] Retrieval            ← Similarity search lấy 4 chunks liên quan nhất
    │
    ▼
[5] Generation           ← Gemini 1.5 Flash tổng hợp câu trả lời
```

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Thư viện / Dịch vụ |
|---|---|
| Web UI | `Streamlit` |
| RAG Framework | `LangChain` |
| PDF Parser | `pypdf` |
| Embedding Model | `Google embedding-001` |
| Vector Database | `ChromaDB` (in-memory) |
| LLM | `Google Gemini 1.5 Flash` |
| Language | `Python 3.10+` |

---

## ⚙️ Hướng dẫn cài đặt

### 1. Clone / tải source về

```bash
git clone <repo-url>
cd NLP_THNN_MS
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Lấy Google Gemini API Key

1. Truy cập [https://aistudio.google.com](https://aistudio.google.com)
2. Đăng nhập bằng tài khoản Google
3. Vào **Get API Key** → **Create API Key**
4. Copy key (bắt đầu bằng `AIza...`)

---

## 🚀 Cách chạy ứng dụng

```bash
streamlit run app.py
```

Trình duyệt tự động mở tại `http://localhost:8501`

### Các bước sử dụng

1. **Nhập API Key** vào ô trong sidebar (trái)
2. **Tải file PDF** lên (ví dụ: giáo trình, slide bài giảng)
3. **Chờ xử lý** (~10-30 giây tuỳ kích thước file)
4. **Đặt câu hỏi** trong khung chat và nhận câu trả lời từ AI

---

## 📁 Cấu trúc dự án

```
NLP_THNN_MS/
├── app.py            # Toàn bộ logic ứng dụng
├── requirements.txt  # Dependencies
└── README.md         # Tài liệu này
```

---

## 💡 Lưu ý

- File PDF nên dưới **50MB** để tốc độ xử lý nhanh.
- API Key **không được lưu** — chỉ dùng trong session hiện tại.
- ChromaDB chạy **in-memory**: dữ liệu mất khi reload trang (phù hợp demo).
- Mô hình hoạt động tốt nhất với PDF **text-based** (không hỗ trợ scan ảnh).
