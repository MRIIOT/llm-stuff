from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
import os
from dotenv import load_dotenv

load_dotenv()


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "pr-ordinary-recreation-28"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)


model = init_chat_model("gpt-4o-mini", model_provider="openai")

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# Define a new graph
workflow = StateGraph(state_schema=State)


# Define the function that calls the model
def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

language = "English"
config = {"configurable": {"thread_id": "abc123"}}

while True:
    try:
        query = input("You: ")
        input_messages = [HumanMessage(query)]
        #output = app.invoke(
        #    {"messages": input_messages, "language": language},
        #    config,
        #)
        #output["messages"][-1].pretty_print()
        for chunk, metadata in app.stream(
                {"messages": input_messages, "language": language},
                config,
                stream_mode="messages"):
            if isinstance(chunk, AIMessage):  # Filter to just model responses
                print(chunk.content, end="")


    except KeyboardInterrupt:
        pass