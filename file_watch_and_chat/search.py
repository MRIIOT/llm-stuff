from pymongo import MongoClient
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

load_dotenv()


ATLAS_DATABASE = "VectorStore"
ATLAS_COLLECTION = "text"

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


client = MongoClient(os.getenv("ATLAS_CONNECTION_STRING"))
collection = client[ATLAS_DATABASE][ATLAS_COLLECTION]
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vector_store = MongoDBAtlasVectorSearch(collection, embeddings, index_name="vector_index")

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
chain = RunnableWithMessageHistory(llm, get_session_history)

while True:
    try:
        user_input = input("You: ")
        docs = vector_store.similarity_search(user_input, k=5)
        context = "\n".join(doc.page_content for doc in docs)
        response = chain.invoke(input=f"Context: {context}\nUser: {user_input}", config={"configurable": {"session_id": "1"}})
        print(f"Bot: {response.content}")

    except KeyboardInterrupt:
        break