#=================================================
# most of the time we use deeplake for Vectordb
#=================================================
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.vectorstores import deeplake, 
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
# API keys should be in environment variables
os.environ["ACTIVELOOP_TOKEN"] = os.getenv('ACTIVELOOP_API_KEY')

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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
my_activeloop_org_id = "sheshpalsingh2024's Org"
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = deeplake(
    dataset_path=dataset_path,
    embedding_function=embeddings,
    # overwrite=True
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