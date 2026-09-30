#=================================================
# Most of the time we use Deep Lake for VectorDB
#=================================================
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. Load environment variables (from local dir and project root)
current_dir = Path(__file__).parent
load_dotenv(current_dir / ".env")
load_dotenv()

# Set ACTIVELOOP_TOKEN for Deep Lake
activeloop_token = (
    os.getenv("ACTIVELOOP_TOKEN")
    or os.getenv("ACTIVELOOP_API_KEY")
    or os.getenv("DEEPLAKE_API_KEY")
    or os.getenv("DEEPLAKE_API_TOKEN")
)
if activeloop_token:
    os.environ["ACTIVELOOP_TOKEN"] = activeloop_token

# 2. Compatibility shim: LangChain 1.x moved retrievers to langchain-classic,
# but langchain-deeplake expects langchain.retrievers
import langchain
import langchain_classic.retrievers
langchain.retrievers = langchain_classic.retrievers
import langchain_classic.retrievers.self_query.base
sys.modules["langchain.retrievers"] = langchain_classic.retrievers
sys.modules["langchain.retrievers.self_query"] = langchain_classic.retrievers.self_query
sys.modules["langchain.retrievers.self_query.base"] = langchain_classic.retrievers.self_query.base

from langchain_deeplake import DeeplakeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use a modern chat model instead of the legacy completion model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# Create Deep Lake dataset
# Activeloop username/org slug (must NOT contain spaces or apostrophes)
my_activeloop_org_id = "sheshpalsingh2024"
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

# Note: You can also use a local path if working offline:
# dataset_path = "./data/deeplake_db"

db = DeeplakeVectorStore(
    dataset_path=dataset_path,
    embedding_function=embeddings,
    token=activeloop_token,
    overwrite=True,
)
db.add_documents(docs)

# Build a retriever (modern vector store API)
retriever = db.as_retriever(search_kwargs={"k": 2})

# Modern RAG: a simple function instead of RetrievalQA
def rag_query(question: str) -> str:
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = (
        "Use the context below to answer the question. "
        "If you don't know, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return llm.invoke(prompt).content

# Run it
print(rag_query("When was Napoleon Bonaparte born?"))