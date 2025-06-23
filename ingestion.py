import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()


if __name__ == '__main__':
    print("Ingesting")
    text_loader = TextLoader("/Users/webshar/Desktop/intro-to-vector-dbs/mediumblog1.txt")
    document = text_loader.load()       # This will load the file in the langchain

    print("Splitting")
    text_splitter = CharacterTextSplitter(chunk_size = 1000,chunk_overlap = 0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    print("Ingesting")

    PineconeVectorStore.from_documents(texts, embeddings,index_name= os.getenv("INDEX_NAME"))
    print("Done")

