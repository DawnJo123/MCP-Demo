from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

async def main():
    client=MultiServerMCPClient(
        {
            "math":{
                "command":"python",
                "args":["mathserver.py"], ## Ensure correct absolute path
                "transport":"stdio",
            
            },
            "weather": {
                "url": "http://localhost:8000/mcp",  # Ensure server is running here
                "transport": "streamable_http",
            }

        }
    )

    os.environ["OPENAI_API_VERSION"] = os.getenv("VERSION")
    os.environ["AZURE_OPENAI_API_KEY"]=os.getenv("AZURE_OPENAI_API")
    os.environ['AZURE_OPENAI_ENDPOINT']=os.getenv("AZURE_ENDPOINT")
    os.environ["OPENAI_MODEL"]=os.getenv("MODEL")

    tools=await client.get_tools()

    model = AzureChatOpenAI(deployment_name=os.environ["OPENAI_MODEL"],
                  api_version=os.environ["OPENAI_API_VERSION"],
                  temperature=0,
                  max_tokens=200,
                  timeout=None,
                  max_retries=2,
    )
    agent= create_react_agent(
        model,tools
    )

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Compute 8-4 using only the given tools"}]}
    )

    print("Math response:", math_response['messages'][-1].content)

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in California?"}]}
    )
    print("Weather response:", weather_response['messages'][-1].content)

asyncio.run(main())
