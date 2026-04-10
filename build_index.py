import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def build_offline_index():
    print("🚀 開始建立向量索引...")
    all_documents = []

    # 1. 處理 FAQ.json
    if os.path.exists("FAQ.json"):
        print("📝 正在讀取 FAQ.json...")
        with open("FAQ.json", "r", encoding="utf-8") as f:
            faq_data = json.load(f)
        for item in faq_data:
            text = f"Q: {item.get('常見問題')}\nA: {item.get('問題答覆')}"
            all_documents.append(Document(page_content=text, metadata={"source": "FAQ.json"}))
    
    # 2. 處理 pdfs 資料夾
    pdf_folder = "pdfs"
    if os.path.exists(pdf_folder):
        print(f"📄 正在讀取 {pdf_folder} 資料夾內的 PDF...")
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
        
        pdf_raw_docs = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
            pdf_raw_docs.extend(loader.load())
        
        if pdf_raw_docs:
            # 這裡進行切片，避免一段文字太長
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            pdf_chunks = text_splitter.split_documents(pdf_raw_docs)
            all_documents.append(pdf_chunks)
            print(f"✅ 已切分出 {len(pdf_chunks)} 個 PDF 區塊")

    if not all_documents:
        print("❌ 錯誤：找不到任何資料來源（FAQ.json 或 pdfs/）")
        return

    # 3. 初始化本地 Embedding 模型
    print("🧠 正在啟動本地模型（這會用到你的 CPU/GPU 運算力）...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 4. 建立索引並存檔
    print("⚡ 正在轉換向量並建立索引，請稍候...")
    # 注意：這裡如果是 list 的 list 要攤平
    flat_docs = []
    for doc in all_documents:
        if isinstance(doc, list): flat_docs.extend(doc)
        else: flat_docs.append(doc)

    vector_db = FAISS.from_documents(flat_docs, embeddings)
    
    # 【關鍵存檔指令】
    vector_db.save_local("faiss_index")
    
    print("\n" + "="*30)
    print("🎉 大功告成！")
    print(f"已成功產出資料夾：{os.path.join(os.getcwd(), 'faiss_index')}")
    print("裡面應該包含：index.faiss 與 index.pkl")
    print("="*30)

if __name__ == "__main__":
    build_offline_index()