from langchain_core.messages import AIMessage, ToolMessage
from agent import build_agent

def chat():
    agent = build_agent()
    config = {"configurable": {"thread_id": "pdf-session"}}
    print("PDF Agent hazır! 'quit' ile çık.\n")

    while True:
        user_input = input("Sen: ").strip()
        if not user_input or user_input.lower() == "quit":
            break

        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        for msg in response["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  📖 [{tc['name']}] → {tc['args']}")
            if isinstance(msg, ToolMessage):
                print(f"  ↳ {msg.content[:120]}...")

        print(f"Agent: {response['messages'][-1].content}\n")

if __name__ == "__main__":
    chat()


