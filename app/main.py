import asyncio

from client import McpClient
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()


class Agent:
    def __init__(self, model: str = "gpt-3.5-turbo", temperature=0.5):
        self.model = ChatOpenAI(model=model)
        self.workflow = self._build_workflow()
        self.client = McpClient()

    def _build_workflow(self):
        graph = StateGraph(MessagesState)

        graph.add_node("agent", self.call_agent)
        graph.add_node("tool_execution", self.tool_execution)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tool_execution": "tool_execution",
                "end": END,
            },
        )
        graph.add_edge("tool_execution", END)
        return graph.compile()

    async def call_agent(self, state: MessagesState):
        system_message = SystemMessage(
            content="""
            Kamu adalah asisten pribadi yang membantu pengguna dengan pertanyaan mereka.
            Tugas kamu adalah memahami pertanyaan pengguna dan memberikan jawaban yang relevan.

            Gunakan tools yang tersedia untuk menjawab pertanyaan pengguna.
            Jika kamu tidak dapat menjawab pertanyaan, beritahu pengguna bahwa kamu tidak tahu jawabannya.

            Berikut adalah daftar tools yang tersedia:
            - read_document: 
                - Membaca dokumen dan memberikan informasi yang relevan.
                - args:
                    - query: Pertanyaan dari pengguna.
                    query harus berupa pertanyaan.
            
        """
        )
        await self.client.connect_to_server()
        messages = state["messages"]
        tools = await self.client.get_tools()

        llm = self.model.bind_tools(tools)
        response = await llm.ainvoke([system_message] + messages)
        print("LLM Response:", response.content)

        return {"messages": messages + [response]}

    def should_continue(self, state: MessagesState):
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tool_execution"
        return "end"

    async def tool_execution(self, state: MessagesState):
        # await self.client.connect_to_server()
        last_message = state["messages"][-1]
        tool_messages = []
        if last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                result = await self.client.session.call_tool(
                    tool_call["name"], arguments=tool_call["args"]
                )
                tool_messages.append(
                    ToolMessage(content=result, tool_call_id=tool_call["id"])
                )
        return {"messages": state["messages"] + tool_messages}

    async def run(self, user_query: str):
        messages = {"messages": [HumanMessage(content=user_query)]}
        result = await self.workflow.ainvoke(messages)

        return result


async def main():
    agent = Agent()
    user_query = "siapakah khaerul lutfi?"
    result = await agent.run(user_query)
    print("Final Result:", result)


if __name__ == "__main__":
    asyncio.run(main())
