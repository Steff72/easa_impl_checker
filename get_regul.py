from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone

import pinecone

pinecone.init(api_key="9b1de6ea-8e1a-4e9a-ab8a-63c3bf5f1f37", environment="gcp-starter")


# loader = PyPDFLoader('easa regul/Easy_Access_Rules_for_Aircrew__Regulation__EU__No_1178-2011__-_Revision_from_August_2023.pdf')
# documents = loader.load()
# print(f"{len(documents)} loaded")

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# doc_split = text_splitter.split_documents(documents)
# print(f"{len(doc_split) } documents after split")

embeddings = OpenAIEmbeddings()
# Pinecone.from_documents(doc_split, embeddings, index_name="easa-regulation")
# print("Docs added to Pinecone vectorstore vectors")

docsearch = Pinecone.from_existing_index(
    index_name="easa-regulation", embedding=embeddings
)


qa_chain = RetrievalQA.from_chain_type(
    llm = ChatOpenAI(verbose=True, temperature=0, model="gpt-4-1106-preview"),
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)


result = qa_chain({'query': 'Summarize paragraph FCL.040, inluding the AMC and GM part.'})
print(result)