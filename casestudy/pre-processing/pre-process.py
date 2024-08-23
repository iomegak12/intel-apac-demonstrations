import os

from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv


def get_pdf_text(pdf_document):
    text = ""
    pdf_reader = PdfReader(pdf_document)

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


def create_documents(pdf_files):
    documents = []

    for file in pdf_files:
        chunks = get_pdf_text(file)

        documents.append(
            Document(
                page_content=chunks,
                metadata={
                    "source": file,
                    "type": "PDF",
                    "owner": "Ramkumar JD"
                }
            )
        )

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

        index_name = "intel-apac-index"
        directory_path = "./Docs"
        files = os.listdir(directory_path)
        pdf_files = []

        for file in files:
            pdf_file = directory_path + "/" + file
            pdf_files.append(pdf_file)

            print(f"Processing Required ... File: PDF File {pdf_file}")

        documents = create_documents(pdf_files)
        embeddings = create_embeddings()

        push_documents_to_pinecone(index_name=index_name,
                                   embeddings=embeddings,
                                   documents=documents)

        print("Vector Database and Embeddings are processing successfully ..")
    except Exception as e:
        print(f"Error Occurred, Details : {e}")


if __name__ == "__main__":
    main()
