import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")


def ingest_docs() -> None:
    sources = [
        "EDW Manuals/EDW CSPM Rev37.pdf",
        "EDW Manuals/EDW OM A Rev51.pdf",
        "EDW Manuals/EDW OM B A320 Rev15.pdf",
        "EDW Manuals/EDW OM B A340 Rev19.pdf",
        "EDW Manuals/EDW OM C Initial Issue.pdf",
        "EDW Manuals/EDW OM D Rev29.pdf",
        "EDW Manuals/EDW OMM Issue 2 Rev02 - Draft 1.pdf",
    ]
    documents = []

    for source in sources:
        loader = PyPDFLoader(source)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc_split = text_splitter.split_documents(documents)
    print(f"loaded {len(doc_split) } documents")

    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(doc_split, embeddings, index_name="edw-doc-index")
    print("Docs added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()
