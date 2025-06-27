import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chains.retrieval import create_retrieval_chain  # class which will do retrieval argumentation
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings,GoogleGenerativeAI

INDEX_NAME = "langchain-doc-index"

def llm_run(query:str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    docsearch = PineconeVectorStore(index_name=INDEX_NAME,embedding=embeddings)  #It holds the functionality of similarity search
    google = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        verbose=True
    ) # This is our llm model.

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_documents_chain = create_stuff_documents_chain(google,retrieval_qa_chat_prompt)
    qa = create_retrieval_chain(
        retriever = docsearch.as_retriever(),combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input":query})
    return result


if __name__ == "__main__":
    res = llm_run("What is langchain chian?")
    print(res["answer"])