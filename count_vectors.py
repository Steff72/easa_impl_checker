import os
import pinecone

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")


index = pinecone.Index("edw-doc-index")

print(index.describe_index_stats())
