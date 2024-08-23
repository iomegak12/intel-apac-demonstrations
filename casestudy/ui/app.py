import os
import streamlit as st

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.summarize import load_summarize_chain


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


def get_summary_llm(llm, current_document):
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_document])

    return summary


def main():
    try:
        load_dotenv()
        openai_api_key = os.environ["OPENAI_API_KEY"]

        llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0.5,
            openai_api_key=openai_api_key
        )

        index_name = "intel-apac-index"

        st.set_page_config(page_title="Resume Screening Assistant")
        st.title("Resume Screening AI Assistant")
        st.subheader(
            "This AI Assistant would help you to scree available and indexed resumes, that are submitted to the organization!"
        )

        job_descripton = st.text_area("Please pase the 'Job Descripton' here ...",
                                      key="1",
                                      height=200)

        document_count = st.text_input("No. Of Resume(s) To Return", key="2")

        submit = st.button("Analyze")

        if submit:
            embeddings = create_embeddings()
            relevant_documents = search_similar_documents(
                job_descripton, int(document_count), index_name=index_name, embeddings=embeddings)

            for document_index in range(len(relevant_documents)):
                st.subheader(":sparkles:" + str(document_index+1))

                file_name = "** FILE ** " + \
                    relevant_documents[document_index].metadata["source"]

                st.write(file_name)

                with st.expander("Show Me Summary ... :heavy_plus_sign:"):
                    summary = get_summary_llm(
                        llm=llm, current_document=relevant_documents[document_index])

                    st.write("*** SUMMARY *** " + summary)
    except Exception as e:
        print(f"Error Occurred, Details : {e}")
        raise


if __name__ == "__main__":
    main()
