import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings,GoogleGenerativeAI
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_pinecone import PineconeVectorStore

embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def ingest_doc():
    loader = ReadTheDocsLoader("/Users/webshar/Desktop/documentation-helper/langchain-docs")
    raw_document = loader.load()
    print(f"loaded {len(raw_document)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=550,chunk_overlap=50)
    documents = text_splitter.split_documents(raw_document)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents,embeddings,index_name = "langchain-doc-index")

    print("****Loading vectorstore done *****")


if __name__ == "__main__":
    print("Ingestion started...")
    ingest_doc()