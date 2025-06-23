import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub  # which will download the retrival prompt
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("Retrieving..")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    query = " what is pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input = {})
    # print(result.content)

    vector_stores = PineconeVectorStore(index_name=os.environ["INDEX_NAME"],embedding=embeddings)

    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval_qa_chat")

    retrieval_qa_chat_prompt = PromptTemplate.from_template(
        """
        Use the following context to answer the user's question. If you don't know the answer, just say you don't know.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """
    )

    combine_docs_chain = create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vector_stores.as_retriever(),combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input = {"input":query})
    print(result)

    template = """ Use the following pieses of context to answer the user's question at the end.
    If you don't know the answer, just say you don't know,don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at end of answer.
    
    {context}
    
    
    Question:{question}
    
    
    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template=template)

    rag_chain = (
        {
            "context": vector_stores.as_retriever() | format_docs, "question": RunnablePassthrough()
        }
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)
    print(res)