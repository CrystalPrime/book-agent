from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import load_all_tools
import os
from dotenv import load_dotenv


load_dotenv()

def build_agent():
    tools = load_all_tools()

    if not tools:
        raise ValueError("Hiç tool yüklenmedi. Önce ingest.py çalıştır.")

    # Tool isimlerini system prompt'a yaz — agent neyin ne olduğunu bilsin
    tool_names = "\n".join([f"- {t.name}" for t in tools])
    model1 = "llama-3.3-70b-versatile"
    #model1 = "openai/gpt-oss-20b"
    #model="openai/gpt-oss-120b"
    llm = ChatGroq(
        model=model1,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=f"""You are a helpful assistant that answers questions 
using the following book sources:
{tool_names}

Always search the most relevant book for the question.
If the question spans multiple books, search each one.
Cite the page number in your answer when available.
Answer in the user's language.""",
        checkpointer=MemorySaver(),
    )