import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


def create_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    return embeddings


def search_similar_documents(query, no_of_results, index_name, embeddings):
    vectore_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    similar_documents = vectore_store.similarity_search(
        query, k=no_of_results)

    return similar_documents


def main():
    try:
        load_dotenv()

        index_name = os.environ["PINECONE_INDEX"]
        embeddings = create_embeddings()
        no_of_results = 2

        query = """
                how to create an email signature
            """

        relevant_documents = search_similar_documents(
            query, no_of_results, index_name, embeddings)

        for doc_index in range(len(relevant_documents)):
            document = relevant_documents[doc_index]

            print(document.metadata["source"])
            print(document.page_content)
            print(document.metadata["owner"])
            print("\n")
    except Exception as e:
        print(f"Error Occurred, Details : {e}")


if __name__ == "__main__":
    main()
