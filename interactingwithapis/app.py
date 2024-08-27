import os

from dotenv import load_dotenv
from langchain.chains import APIChain
from langchain_openai import ChatOpenAI


def main():
    load_dotenv()
    model_name = "gpt-3.5-turbo-0125"
    openai_api_key = os.environ["OPENAI_API_KEY"]

    llm = ChatOpenAI(temperature=-0, model=model_name,
                     openai_api_key=openai_api_key)

    api_documentation = """
        BASE URL: https://restcountries.com/

        API Documentation:

        The API endpoint /v3.1/name/{name} Used to find informatin about a country. All URL parameters are listed below:
            - name: Name of country - Ex: italy, france

        The API endpoint /v3.1/currency/{currency} Uesd to find information about a region. All URL parameters are listed below:
            - currency: 3 letter currency. Example: USD, COP

        Woo! This is my documentation
    """

    chain = APIChain.from_llm_and_api_docs(
        llm,
        api_documentation,
        verbose=True,
        limit_to_domains=None
    )

    response = chain.run("Can you tell me information about france?")

    print(response)

    response = chain.run("Can you tell me about the currency COP?")

    print(response)


if __name__ == "__main__":
    main()
