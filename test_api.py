import google.generativeai as genai

# Em nhớ dán cái API key thật của em vào trong ngoặc kép nhé
genai.configure(api_key="AIzaSyAfzBZf0HqK0q7KtMVVVvFIbUYZ6Dpjx7U")

print("Danh sách các mô hình mà API Key của em được phép gọi:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print("Lỗi hệ thống hoặc API Key bị khoá:", e)