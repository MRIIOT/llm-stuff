from pymongo import MongoClient
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DedocAPIFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredImageLoader


import time
import os
import shutil
import pymupdf
import traceback
from dotenv import load_dotenv
# pdfminer.six, pi_heif, unstructured, unstructured_inference, unstructured_pytasseract

load_dotenv()

DEDOC_URI = "http://192.168.111.33:1231"
ATLAS_DATABASE = "VectorStore"
ATLAS_COLLECTION = "text"
WATCH_FOLDER = "/Users/chrismisztur/Downloads/watch"
PROCESS_IMAGES = False


def main():
    try:
        print("running...")
        while True:
            for doc_filename in os.listdir(WATCH_FOLDER):
                try:
                    file_path = os.path.join(WATCH_FOLDER, doc_filename)
                    images_path = f"{file_path}_images"
                    print("---")
                    print(f"\tprocessing file '{file_path}'")
                    os.mkdir(images_path)

                    # extract all images in pdf
                    print(f"\t\textracting images")
                    doc = pymupdf.open(file_path)
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        images = page.get_images(full=True)
                        for img_index, img in enumerate(images):
                            base_image = doc.extract_image(img[0])
                            image_filename = f"{images_path}/page{page_num + 1}_img{img_index + 1}.{base_image['ext']}"
                            with open(image_filename, "wb") as f: f.write(base_image["image"])

                    image_data = []
                    if PROCESS_IMAGES is True:
                        # extract text from images
                        print(f"\t\textracting text from images")
                        for img_filename in os.listdir(images_path):
                            image_path = os.path.join(images_path, img_filename)
                            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                            loader = UnstructuredImageLoader(image_path, mode="single", strategy="fast")
                            try:
                                image_data.extend(loader.load_and_split(text_splitter=splitter))
                            except:
                                pass

                    # clean image data
                    image_data = [obj for obj in image_data if len(obj.page_content) >= 100]

                    # extract text from document
                    print(f"\t\textracting text from document")
                    loader = DedocAPIFileLoader(file_path=file_path, url=DEDOC_URI, split="page", need_pdf_table_analysis=True)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    doc_data = loader.load_and_split(text_splitter=splitter)

                    # clean doc data
                    doc_data = [obj for obj in doc_data if len(obj.page_content) >= 100]

                    # get embeddings
                    print(f"\t\tstoring data in vector store")
                    all_data = image_data + doc_data
                    client = MongoClient(os.getenv("ATLAS_CONNECTION_STRING"))
                    collection = client[ATLAS_DATABASE][ATLAS_COLLECTION]
                    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
                    vector_store = MongoDBAtlasVectorSearch.from_documents(all_data, embeddings, collection=collection)

                except Exception as ex:
                    print(f"\t\tERROR processing!")
                    print(f"\t\t\t{ex}")
                    print(f"\t\t\t{traceback.format_exc()}")

                finally:
                    print(f"\t\tdelete file")
                    os.remove(file_path)
                    print(f"\t\tdelete folder")
                    shutil.rmtree(images_path)
                    print(f"\tprocessing complete")

            time.sleep(10)
    except KeyboardInterrupt:
        print(f"terminating!")


main()