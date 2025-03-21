from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import MergedDataLoader
from langchain_community.document_loaders.youtube import YoutubeLoader, TranscriptFormat
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from embeddings import LMStudioEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

LMS_URI = "http://localhost:1234/v1"
EMBED_MODEL = "lmstudio-community/text-embedding-granite-embedding-278m-multilingual"
EMBED_DIMENSIONS = 768
YT_URI1 = "https://www.youtube.com/watch?v=hiE7j1Tvx3Q"
YT_URI2 = "https://www.youtube.com/watch?v=hQKRFg4i3ds"

# LangSmith Tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


llm = ChatOpenAI(openai_api_key="none", base_url=LMS_URI, temperature=0.0, streaming=True)
embeddings = LMStudioEmbeddings(api_url=LMS_URI, model=EMBED_MODEL, dimensions=EMBED_DIMENSIONS)

print("--- INGEST ---")
loader1 = YoutubeLoader.from_youtube_url(YT_URI1, transcript_format=TranscriptFormat.CHUNKS, chunk_size_seconds=30)
loader2 = YoutubeLoader.from_youtube_url(YT_URI2, transcript_format=TranscriptFormat.CHUNKS, chunk_size_seconds=30)
loaderX = MergedDataLoader(loaders=[loader1, loader2])
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
data = loaderX.load_and_split()
vector_store = InMemoryVectorStore.from_documents(data, embeddings)

print("--- SEARCH ---")
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

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
                                              "If answers are not found in conversation history, then you should reference the context provided in your answers."
                                              "Always explain your reasoning."
                                              ))

        current_message = HumanMessage(content=
                                         f"Question: {user_input}\n\n"
                                         f"Context:\n{context}\n\n"
                                         f"Answer:"
                                         )

        full_messages = history.messages + [current_message]

        print("Bot: ", end="", flush=True)
        collected_message = ""

        for chunk in llm.stream(full_messages):
            if hasattr(chunk, "content"):
                token = chunk.content
            elif isinstance(chunk, dict) and "content" in chunk:
                token = chunk["content"]
            else:
                continue

            print(token, end="", flush=True)
            collected_message += token

        print()

        history.add_message(current_message)
        history.add_message(SystemMessage(content=collected_message))

    except KeyboardInterrupt:
        break