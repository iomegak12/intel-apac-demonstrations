import os
import requests

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from bs4 import BeautifulSoup


def access_service_now_kbs():
    service_now_base_uri = os.environ["SERVICE_NOW_BASE_URI"]
    user_name = os.environ["SERVICE_NOW_USER"]
    password = os.environ["SERVCIE_NOW_PASSWORD"]
    credentials = (user_name, password)
    service_now_kb_uri = f"{service_now_base_uri}?sysparm_limit=10"
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.get(service_now_kb_uri,
                            auth=credentials, headers=headers)

    status = response.status_code
    parsed_articles = []

    if status == 200:
        print("Authentication is Successful ...")

        response = response.json()
        articles = response["result"]

        for article in articles:
            parsed_articles.append(article)

    else:
        raise ("Authentication Failed!")

    return parsed_articles


def fetch_and_process_html_text(text, title):
    soup = BeautifulSoup(text, features="html.parser")

    document = Document(page_content=soup.text,
                        metadata={
                            "source": title,
                            "type": "HTML",
                            "owner": "Ramkumar JD"
                        })

    return document


def create_documents(articles):
    documents = []

    for article in articles:
        document = fetch_and_process_html_text(
            text=article["text"],
            title=article["short_description"]
        )

        documents.append(document)

    return documents


def create_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    return embeddings


def push_documents_to_pinecone(index_name, embeddings, documents):
    vectore_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    vectore_store.add_documents(documents)


def main():
    try:
        load_dotenv()

        index_name = os.environ["PINECONE_INDEX"]
        articles = access_service_now_kbs()
        documents = create_documents(articles=articles)
        embeddings = create_embeddings()

        push_documents_to_pinecone(index_name=index_name,
                                   embeddings=embeddings,
                                   documents=documents)

        print("Vector Database and Embeddings are processing successfully ..")
    except Exception as e:
        print(f"Error Occurred, Details : {e}")
        raise


if __name__ == "__main__":
    main()
