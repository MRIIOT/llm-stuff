import asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from helper import get_embedding
import os
from dotenv import load_dotenv

load_dotenv()

ATLAS_DATABASE = "VectorStore"
ATLAS_COLLECTION = "text"
QUERY_TEXT = "How do I mount a samsung TV?"


async def main():
    embedding = await get_embedding(QUERY_TEXT, os.getenv("OPENAI_API_KEY"))

    db_client = AsyncIOMotorClient(
            os.getenv("ATLAS_CONNECTION_STRING"),
            maxPoolSize=5,
            minPoolSize=1,
            uuidRepresentation="standard",
        )

    db: AsyncIOMotorDatabase = db_client[ATLAS_DATABASE]
    result = db[ATLAS_COLLECTION].aggregate([
        {
            "$vectorSearch" : {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": 10
            }
        },
        {
            "$project": {
                "file": 1,
                "page": 1,
                "text": 1
            }
        }
    ])

    async for document in result:
        print(document)


asyncio.run(main())