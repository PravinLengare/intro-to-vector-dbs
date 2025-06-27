from typing import Set

from backend.core import llm_run
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.header("Retrieval QA Chat Prompt")

INDEX_NAME = "streamlit"
prompt = st.text_input("prompt",placeholder="enter your prompt")
if 'user_prompt_history' not in st.session_state:
    st.session_state.user_prompt_history = []

if 'chat_prompt_history' not in st.session_state:
    st.session_state.chat_prompt_history = []


def create_sources_string(source_urls:Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i,source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"

    return sources_string


if prompt:
    with st.spinner("Generating Responses..."):
        generated_response = llm_run(prompt)
        if "source_documents" in generated_response:
            sources = set(doc.metadata["source"] for doc in generated_response["source_documents"])
        else:
            sources = set()

        sources = set([doc.metadata["source"]for doc in generated_response["source_documents"]])

        formatted_response =  (
            f"{generated_response['result']} \n\n {create_sources_string((sources))}"
        )

    st.session_state.user_prompt_history.append(prompt)
    st.session_state.chat_prompt_history.append(formatted_response)
    import pprint

    pprint.pprint(generated_response)

if st.session_state["user_prompt_history"]:
    for generated_response,user_query in zip(st.session_state["user_prompt_history"]),st.session_state["chat_prompt_history"]:
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)