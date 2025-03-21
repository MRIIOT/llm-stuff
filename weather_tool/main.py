from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent


LMS_URI = "http://localhost:1234/v1"


def weather_query_tool(city: str):
    """
    Get weather for the specified city.

    Args:
        city (str): Name of the city

    Returns:
        str: A str weather conditions.
    """
    return f"The weather in {city} is sunny."


tools = [
    Tool(
        name="WeatherTool",
        func=weather_query_tool,
        description="Get the weather for a specific city."
    )
]

llm = ChatOpenAI(openai_api_key="none", base_url=LMS_URI, temperature=0.0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
response = agent.run("What is the weather in Chicago?")
print(response)