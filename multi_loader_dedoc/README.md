# MongoDb Vector Search Test

0. Run `dedoc` in Docker.  
   `netsh interface portproxy add v4tov4 listenport=1231 listenaddress=0.0.0.0 connectport=1231 connectaddress=172.24.56.104`
1. Create `VectorStore` database.
2. Create `text` collection.
3. Create vector index:

```
db.text.createSearchIndex(
  "vector_index",
  "vectorSearch",
  {
     "fields": [
        {
           "type": "vector",
           "path": "embedding",
           "numDimensions": 1536,
           "similarity": "cosine"
        }
     ]
  }
);
```

4. Run `ingest.py`.
5. Run `search.py`.

