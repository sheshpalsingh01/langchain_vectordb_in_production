import os
import langchain
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


#===========================================
# Retrieveerrs issues
#===========================================
import sys
import langchain_classic.retrievers

langchain.retrievers = langchain_classic.retrievers
import langchain_classic.retrievers.self_query.base

sys.modules["langchain.retrievers"] = langchain_classic.retrievers
sys.modules["langchain.retrievers.self_query"] = langchain_classic.retrievers.self_query
sys.modules["langchain.retrievers.self_query.base"] = (
    langchain_classic.retrievers.self_query.base
)
from langchain_deeplake.vectorstores import DeeplakeVectorStore

load_dotenv()
# set up the LLM and Embedding model
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# =============================================
# LLm model and Embedding model
# =============================================
llm = ChatGroq(model="qwen/qwen3.8-27b")
embedding = OllamaEmbeddings(model="nomic-embed-text:v1.5")


# ==============================================
# Load the Database
# ==============================================
dataset_path = "al://sheshpalsingh2024/langchain_VECTORDB"
# ds = deeplake.open("al://sheshpalsingh2024/langchain_VECTORDB")

# ==============================================
# Retrieved the data
# ==============================================
db = DeeplakeVectorStore(dataset_path=dataset_path, read_only=True)
retriever = db.as_retriever(search_kwargs={"k": 2})


# ==============================================
# Create tool
# ==============================================
@tool
def retrieval_qa(question: str) -> str:
    """Answer questions using information retrieved from documents."""
    response = retriever.invoke({"query": question})
    return response.get("result", str(response))


#================================================
# create the Agent
#================================================
agent = create_agent(
    model=llm,
    tools=[retrieval_qa],
    system_prompt=(
        "You are a helpful assistant. "
        "Use the retrieval_qa_tool to answer questions about documents."
    ),
)

#===============================================
# Run the Agent
#===============================================
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What information is available in the documents?"}
    ]
})
print(response['messages'][-1])