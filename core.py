import os
from typing import Any, List, Dict

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate

from langchain.vectorstores import Pinecone

import pinecone

# initialize pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")

# initialize chatmodel & vectorstore
chat = ChatOpenAI(
    verbose=True, temperature=0, model="gpt-4-1106-preview", model_kwargs={"seed": 123}
)
embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_existing_index(
    index_name="edw-doc-index", embedding=embeddings
)

# define retrievalchain
qa = ConversationalRetrievalChain.from_llm(
    llm=chat,
    retriever=docsearch.as_retriever(search_kwargs={"k": 20}),
    return_source_documents=False,
)

# Access and modify system message in template
for message_template in qa.combine_docs_chain.llm_chain.prompt.messages:
    if isinstance(message_template, SystemMessagePromptTemplate):
        new_template = "For each user-provided regulation, your task is to analyze the provided context documents to determine where and how the regulation is semantically implemented. You are to search through the documents, identify relevant sections, and provide a summary of each occurrence in the format: 'Document: *name of the document, where the occurence was found. [CSPM, OM A, OM B A320, OM B A340, OM C, OM D, OMM]*,\n Section: *section number, eg. 2.5.2.1. only provide the number, not the title*,\n Summary: *Brief Summary of the section*'. The summary should be concise yet informative, capturing the key aspects of how the regulation is addressed in that section. If a regulation is not implemented in any part of the documents, respond with 'The regulation is not implemented in the provided documents.' Your analysis should be thorough, ensuring all potential occurrences are identified and summarized accurately.\n----------------\n{context}"
        message_template.prompt.template = new_template
        break


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    return qa({"question": query, "chat_history": chat_history})["answer"]


if __name__ == "__main__":
    easa_text = "the holder of an instructor certificate may log as PIC all flight time during which he or she acts as an instructor in an aircraft;"

    print(run_llm(easa_text))
