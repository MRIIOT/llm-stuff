# MongoDb Vector Search Test 1

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

