import os
from typing import Any, List, Dict

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone

import pinecone

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")

# chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-3.5-turbo-1106")
chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-4-1106-preview", model_kwargs={'seed': 111})
embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_existing_index(
    index_name="edw-doc-index", embedding=embeddings
)


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(search_kwargs={"k": 20}),
        return_source_documents=False,
    )

    promt = f"Please conduct a detailed analysis of the context for compliance with the following regulations: [{query}]. It's crucial to provide specific section and/or paragraph numbers for each instance where a regulation is mentioned, addressed, or implied, even if they are in the same section as a previous instance. For every instance found, give a detailed reference in the format: 'Document Name, Section, Summary'. Ensure that both section and paragraph numbers are included for each instance. If a regulation is mentioned multiple times in the same section, list each occurrence separately with its specific paragraph number."

    return qa({"question": promt, "chat_history": chat_history})["answer"]


if __name__ == "__main__":
    easa_text = "the holder of an instructor certificate may log as PIC all flight time during which he or she acts as an instructor in an aircraft;"
    print("*" * 60)
    print(run_llm(easa_text))
