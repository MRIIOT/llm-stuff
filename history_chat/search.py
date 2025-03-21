from pymongo import MongoClient
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
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
        history = get_session_history("1")

        if not history.messages:
            history.add_message(SystemMessage(content=
                                              "You are a helpful assistant that answers questions based on retrieved documents. "
                                              "You should prioritize information from the conversation history over any new assumptions or context."
                                              "If answers are not found in conversation history, then you should reference the context provided in your answers and explain your reasoning."
                                              ))

        history.add_message(HumanMessage(content=
                                         f"I have a question: {user_input}\n\n"
                                         f"Here is new relevant context to help you answer:\n{context}"
                                         ))

        response = llm.invoke(history.messages)

        history.add_message(response)
        print(f"Bot: {response.content}")

    except KeyboardInterrupt:
        break