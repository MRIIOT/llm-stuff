from pymongo import MongoClient
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DedocAPIFileLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import MergedDataLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.youtube import YoutubeLoader, TranscriptFormat
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
import os
from dotenv import load_dotenv

load_dotenv()

DEDOC_URI = "http://192.168.111.33:1231"
ATLAS_DATABASE = "VectorStore"
ATLAS_COLLECTION = "text"
FILE = "../mongo_vector_search_manual/fus_810348_manual.pdf"

client = MongoClient(os.getenv("ATLAS_CONNECTION_STRING"))
collection = client[ATLAS_DATABASE][ATLAS_COLLECTION]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

loader1 = DedocAPIFileLoader(file_path=FILE, url=DEDOC_URI, split="page", need_pdf_table_analysis=True)
loader2 = PyMuPDFLoader(file_path=FILE, extract_tables="markdown", mode="single")
loader3 = YoutubeLoaderDL(video_id="taX5-Zln4kM", add_video_info=True)
loader4 = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=l10DhymdeOw", transcript_format=TranscriptFormat.CHUNKS, chunk_size_seconds=30)
loader5 = WikipediaLoader(query="langchain")
loader6 = SeleniumURLLoader(["https://mriiot.com/sharc"])
loaderX = MergedDataLoader(loaders=[loader6])
data = loaderX.load_and_split(text_splitter=splitter)

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorStore = MongoDBAtlasVectorSearch.from_documents(data, embeddings, collection=collection)