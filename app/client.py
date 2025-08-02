import asyncio
from contextlib import AsyncExitStack
from typing import Any, Optional

import nest_asyncio
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

nest_asyncio.apply()


class McpClient:
    def __init__(self, server_path: str = "server.py") -> None:
        self.session: Optional[ClientSession] = None
        self.write: Optional[Any] = None
        self.read: Optional[Any] = None
        self.server_path = server_path
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self):
        server_params = StdioServerParameters(command="python", args=[self.server_path])
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.read, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.read, self.write)
        )

        await self.session.initialize()

        list_tools = await self.session.list_tools()
        print("List tools in server:")
        for tool in list_tools.tools:
            print(f"- {tool.name} : {tool.description}")

    async def get_tools(self):
        if not self.session:
            raise ValueError("Session not initialized. Call connect_to_server first.")
        list_tools = await load_mcp_tools(self.session)
        if not list_tools:
            raise ValueError("No tools found in the session.")

        print("Successfully getting tools.")
        return list_tools

    async def tool_execution(self, tool_name: str, query: str):
        print(type(self.session))
        if not self.session:
            raise ValueError("Session not initialized. Call connect_to_server first.")
        result = await self.session.call_tool(tool_name, {"query": query})
        return result


if __name__ == "__main__":

    async def main():
        client = McpClient()
        await client.connect_to_server()
        await client.get_tools()
        tool_call = await client.tool_execution("read_document", "What is my name?")
        print(f"Tool call result: {tool_call}")

    asyncio.run(main())
