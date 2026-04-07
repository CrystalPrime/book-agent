from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
import os

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def make_book_tool(book_name: str, index_path: str):
    db = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    @tool
    def search_book(query: str) -> str:
        """Search the book for relevant information based on the query."""
        docs = db.similarity_search(query, k=2)
        if not docs:
            return f"No relevant content found in {book_name}."
        results = []
        for doc in docs:
            page = doc.metadata.get("page", "?")
            results.append(f"[{book_name} — p.{page}]\n{doc.page_content[:300]}")
        return "\n\n".join(results)

    search_book.__name__ = f"search_{book_name}"
    return search_book

def load_all_tools():
    tools = []
    if not os.path.exists("indexes"):
        return tools
    for index_name in os.listdir("indexes"):
        index_path = f"indexes/{index_name}"
        if os.path.isdir(index_path):
            tool_fn = make_book_tool(index_name, index_path)
            tools.append(tool_fn)
            # print(f"[tool loaded] search_{index_name}")
    return tools