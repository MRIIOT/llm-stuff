from pymongo import MongoClient
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai.llms import OpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

ATLAS_DATABASE = "VectorStore"
ATLAS_COLLECTION = "text"
QUERY_TEXT = "What is a SHARC?"

client = MongoClient(os.getenv("ATLAS_CONNECTION_STRING"))
collection = client[ATLAS_DATABASE][ATLAS_COLLECTION]
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorStore = MongoDBAtlasVectorSearch(collection, embeddings, index_name="vector_index")

def query_data(query):
    docs = vectorStore.similarity_search(query, k=1)
    as_output = docs[0].page_content
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    retriever = vectorStore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    retriever_output = qa.invoke(query)
    print(as_output)
    print("---")
    print(retriever_output)
    return as_output, retriever_output

a, b = query_data(QUERY_TEXT)