from langchain.chat_models import ChatOpenAI
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import SystemMessagePromptTemplate

from langchain.vectorstores import Pinecone

import pinecone

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")

chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-3.5-turbo-1106")
embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_existing_index(
    index_name="edw-doc-index", embedding=embeddings
)

qa = ConversationalRetrievalChain.from_llm(
    llm=chat,
    retriever=docsearch.as_retriever(search_kwargs={"k": 20}),
    return_source_documents=False,
)

for message_template in qa.combine_docs_chain.llm_chain.prompt.messages:
    if isinstance(message_template, SystemMessagePromptTemplate):
        # Access and modify the template
        new_template = "Your new template string here"
        message_template.prompt.template = new_template
        break

print(qa.combine_docs_chain.llm_chain.prompt.messages)
