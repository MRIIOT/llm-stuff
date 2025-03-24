# https://www.youtube.com/watch?v=vWVJgY-MLcc

from os import getenv
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, PromptTemplate
from langchain_community.vectorstores import FAISS, MongoDBAtlasVectorSearch
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

load_dotenv()

synthetic_data = [

]

system_prompt = '''
You are an expert in MongoDb queries. Your primary task is to generate accurate, efficient and syntactically correct MongoDb queries based on the provided schema and examples.
Your job is to interpret user's query and generate the corresponding MongoDb queries to retrieve data from the collection.
Here is the user query:
{user_query}

Take reference from below attached example when answering the query:
'''

embeddings = OpenAIEmbeddings(openai_api_key=getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(openai_api_key=getenv("OPENAI_API_KEY"), temperature=0)
selector = SemanticSimilarityExampleSelector.from_examples(
    synthetic_data,
    embeddings,
    FAISS,
    k=3,
    input_keys=["user_query"]
)
few_shot = FewShotChatMessagePromptTemplate(
    example_selector=selector,
    example_prompt=ChatPromptTemplate.from_messages(
        [
            ("human", "{user_query}"),
            ("ai", "{mongo_query}")
        ]
    ),
    input_variables=["user_query"]
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        few_shot,
        ("human", "{user_query}")
    ]
)
chain = prompt | llm

response = chain.invoke("{user_query}", "find all notes where Job Number is equal to test")
print(response.content)
