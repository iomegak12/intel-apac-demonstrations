from decouple import config
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes

app = FastAPI()

model = ChatOpenAI(
    openai_api_key=config("OPENAI_API_KEY"),
    model="gpt-3.5-turbo-0125"
)

prompt = ChatPromptTemplate.from_template(
    "Give me a summary about {topic} in a paragraph or less")

chain = prompt | model


add_routes(app, chain, path="/openai")

if __name__ == "__main__":
    import uvicorn

    try:
        uvicorn.run(app, host='0.0.0.0', port=8000)
    except Exception as error:
        print(f"Error Occurred, Details : {error}")

        raise
