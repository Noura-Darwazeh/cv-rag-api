# ═══════════════════════════════════════════════════
# 1️⃣ Imports
# ═══════════════════════════════════════════════════
import os
from typing import List

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# ═══════════════════════════════════════════════════
# 2️⃣ Config
# ═══════════════════════════════════════════════════
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"
PINECONE_INDEX   = "cv-rag"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
RETRIEVER_TOP_K  = 5

# ═══════════════════════════════════════════════════
# 3️⃣ Globals
# ═══════════════════════════════════════════════════
pc = None
index = None
embed_model = None
retriever = None
rag_chain = None

print("Imports OK ✅")


# ═══════════════════════════════════════════════════
# 4️⃣ Retriever
# ═══════════════════════════════════════════════════
class PineconeRetriever(BaseRetriever):
    index: object
    embed_model: object
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).tolist()[0]

        response = self.index.query(
            vector=query_vector,
            top_k=self.top_k,
            include_metadata=True
        )

        docs = []
        for match in response["matches"]:
            docs.append(
                Document(
                    page_content=match["metadata"].get("text", ""),
                    metadata={
                        "doc_name": match["metadata"].get("doc_name", "unknown"),
                        "score": round(match["score"], 4),
                    },
                )
            )
        return docs


# ═══════════════════════════════════════════════════
# 5️⃣ Helpers
# ═══════════════════════════════════════════════════
def format_docs(docs: List[Document]) -> str:
    return "\n---\n".join(
        f"[{d.metadata.get('doc_name','unknown')} | {d.metadata.get('score','?')}]\n{d.page_content}"
        for d in docs
    )


# ═══════════════════════════════════════════════════
# 6️⃣ Init RAG
# ═══════════════════════════════════════════════════
def init_rag():
    global pc, index, embed_model, retriever, rag_chain

    print("Initializing RAG... 🚀")

    # Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    print("Connected to Pinecone ✅")

    # Embeddings
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded ✅")

    retriever = PineconeRetriever(
        index=index,
        embed_model=embed_model,
        top_k=RETRIEVER_TOP_K,
    )

    # Prompt (IMPORTANT FIX)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a CV assistant.\n"
         "Answer ONLY using the context below.\n\n"
         "{context}"),
        ("human", "{question}")
    ])

    # LLM
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model_name=OPENROUTER_MODEL,
        temperature=0.2,
    )

    # Retriever wrapper
    def retrieve(q: str):
        docs = retriever._get_relevant_documents(q)
        return format_docs(docs)

    # RAG CHAIN (FIXED)
    rag_chain = (
        {
            "context": RunnableLambda(retrieve),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG ready ✅")


# ═══════════════════════════════════════════════════
# 7️⃣ API functions
# ═══════════════════════════════════════════════════
def run_query(question: str, show_context: bool = False):
    docs = retriever._get_relevant_documents(question)
    context = format_docs(docs)

    answer = rag_chain.invoke(question)

    return {
        "question": question,
        "answer": answer,
        "context": context if show_context else None,
    }


def run_query_stream(question: str):
    for chunk in rag_chain.stream(question):
        yield chunk