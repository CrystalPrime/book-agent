from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ yeni
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings
import os

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
)

def ingest_pdf(pdf_path: str):
    name = os.path.splitext(os.path.basename(pdf_path))[0]
    index_path = f"indexes/{name}"

    if os.path.exists(index_path):
        print(f"[skip] {name} zaten mevcut")
        return

    print(f"[ingest] {name} işleniyor...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    chunks = splitter.split_documents(pages)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_path)
    print(f"[done] {name} → {len(chunks)} chunk kaydedildi")

if __name__ == "__main__":
    os.makedirs("indexes", exist_ok=True)
    for filename in os.listdir("books"):
        if filename.endswith(".pdf"):
            ingest_pdf(f"books/{filename}")
