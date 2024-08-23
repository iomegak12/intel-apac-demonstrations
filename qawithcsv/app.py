import pandas as pd
import os
import streamlit as st

from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


def main():
    load_dotenv()

    st.set_page_config(
        page_title="CSV Documents - Chat for HR Attritions",
        page_icon=":books:"
    )

    st.title("HR - Attritions Analysis Chatbot")
    st.subheader("Helps to uncover insights from HR Attrition Data!")

    st.markdown(
        """
            This chatbot is created to demonstrate and answer questions from a set of attributes
            data from your CSV file, that was curated by the organization Data Engineering Team!
        """
    )

    user_question = st.text_input(
        "Ask your questions about HR Attrition Data ..."
    )

    csv_path = "./hr-employees-attritions-internet.csv"
    df = pd.read_csv(csv_path)

    model_name = "gpt-3.5-turbo-0125"
    openai_api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(model_name=model_name, temperature=0,
                     openai_api_key=openai_api_key)

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )

    agent.handle_parsing_errors = True

    answer = agent.invoke(user_question)

    st.write(answer["output"])


if __name__ == "__main__":
    main()
