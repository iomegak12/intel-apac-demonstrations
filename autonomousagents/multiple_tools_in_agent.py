import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.utilities import BingSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


@tool("BingSearch")
def search(search_query: str):
    """
    Useful for when you need to search the internet for more information

    Args:

    search_query(str): search query
    """

    search_tool = BingSearchAPIWrapper()
    return search_tool.run(search_query)


@tool()
def add(first_int: int, second_int: int) -> int:
    """
    Adds / sums two integers
    """

    return first_int + second_int


@tool()
def exponentize(base: int, exponent: int) -> int:
    """
    Exponentize the base to the exponent value
    """

    return base ** exponent


@tool()
def multiply(first_int: int, second_int: int) -> int:
    """
    Multiplies two integers
    """

    return first_int * second_int


def main():
    load_dotenv()

    model_name = "gpt-4"
    openai_api_key = os.environ["OPENAI_API_KEY"]

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=5000,
        openai_api_key=openai_api_key
    )

    tools = [add, multiply, exponentize, search]

    template = """
        SYSTEM

        You are a helpful assistant

        PLACEHOLDER
        {chat_history}
        
        HUMAN

        {input}

        PLACEHOLDER
        {agent_scratchpad}
    """

    prompt = PromptTemplate(
        input_variables=["chat_history", "input"],
        template=template
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    agent = create_tool_calling_agent(
        llm,
        tools,
        prompt,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        max_iterations=10,
    )

    question = "take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result. And let me know what's the capital of Malaysia?"

    response = executor.invoke({
        "input": question
    })

    print(response)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error Occurred, Details : {error}")
