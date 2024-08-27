import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory

def main():
    load_dotenv()

    model_name = "gpt-3.5-turbo-0125"
    openai_api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(model_name = model_name,
                     temperature=0,
                     openai_api_key=openai_api_key)
    
    template = """

    You're a chatbot that is helpful, and your goal is to help the user with humorous way.
    Take what the user is saying, and make a joke about it.

    {chat_history}
    Human: {human_input}
    Chatbot:

    """

    prompt = PromptTemplate(
        input_variables = ["chat_history", "human_input"],
        template = template
    )

    memory = ConversationBufferMemory(memory_key = "chat_history")

    chain = LLMChain(
        llm = llm,
        prompt = prompt,
        verbose = True,
        memory = memory
    )

    question = "is an pear a fruit or vegetable?"

    response = chain.predict(human_input = question)
    
    print(response)

    question2 = "what was one the fruits I first asked you about?"

    response2 = chain.predict(human_input = question2)

    print(response2)


if __name__ == "__main__":
    main()