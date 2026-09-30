# Complete Beginner's Guide: Vector Databases, Deep Lake, and Modern RAG with LangChain

This guide assumes **zero prior knowledge**. Whether you are new to AI, vector databases, or LangChain, this document explains every concept from scratch, explains how permissions and tokens work, details why specific errors happen, and walks through the complete code in [`03_vectordb.py`](file:///home/sheshpal/Documents/Activloop/langchain_vectordb_in_Production/src/langchain_vectordb_in_production/01-Langchain-zero-to-hero/02-GetStart/03_vectordb.py).

---

## 1. High-Level Concepts (From First Principles)

### What is an LLM and Why Does it Need External Data?
Large Language Models (like OpenAI's GPT-4o) are trained on vast amounts of public internet data up to a specific cutoff date. However:
1. They **do not know your private documents**, company internal data, or notes.
2. They can **hallucinate** (make up convincing-sounding but false facts).
3. Re-training or fine-tuning them is expensive and slow.

### What is RAG (Retrieval-Augmented Generation)?
Instead of retraining the model, **RAG** gives the model an open-book exam:
1. **Retrieve:** When a user asks a question, search your database for the 2–3 most relevant paragraphs of information.
2. **Augment:** Paste those retrieved paragraphs into the prompt alongside the question.
3. **Generate:** Ask the LLM: *"Based only on the provided context above, answer the question."*

```
User Question ───────► Search VectorDB ───────► Retrieve Top 2 Relevant Passages
                            │                                  │
                            ▼                                  ▼
               Combine (Question + Passages) ────────► Send to LLM ────────► Accurate Answer
```

---

## 2. What is an Embedding and a Vector Database?

### What is an Embedding?
Computers cannot understand raw words like "Napoleon" or "Emperor" directly. An **embedding model** (such as OpenAI's `text-embedding-3-small`) reads a piece of text and converts it into a long list of numbers called a **vector** (e.g., `[0.021, -0.453, 0.812, ...]`, usually 1,536 numbers long).
- Sentences with **similar meanings** end up with vectors that are close to each other in mathematical space.
- Example: The embedding for *"Napoleon was born in 1769"* will be mathematically very close to *"When was Bonaparte born?"*, even though the words are not identical.

### What is a Vector Database (VectorDB)?
Traditional databases (like SQL or MongoDB) search for **exact matches** or keywords. But if a user searches for *"French military ruler birth year"*, a traditional keyword search might fail if the document says *"Napoleon Bonaparte was born in 1769"*.

A **Vector Database** stores these numerical vector embeddings. When you ask a question, the database computes the distance (cosine similarity) between your question's vector and all stored document vectors, instantly returning the most semantically relevant documents.

### What is Activeloop Deep Lake?
**Deep Lake** (developed by Activeloop) is a specialized vector database designed for AI and deep learning:
- It supports text, embeddings, images, audio, and metadata.
- It operates serverless in the cloud (Activeloop Hub) or locally on your hard drive.
- Datasets stored on Activeloop Hub come with a visual web user interface where you can browse and inspect your vectors and texts in a web browser.

---

## 3. Activeloop Accounts, Permissions & API Tokens

To store datasets on the cloud with Deep Lake, your script needs to authenticate with Activeloop's servers.

### 1. The API Token (`ACTIVELOOP_TOKEN`)
- **Where to get it:** Log in at [https://app.activeloop.ai](https://app.activeloop.ai), click your profile icon in the bottom-left corner, and select **API Tokens**.
- **What it looks like:** A long encoded JWT string (e.g., `eyJhbGciOiJIUzUxMiIs...`).
- **Where to save it:** In your project's `.env` file:
  ```env
  ACTIVELOOP_TOKEN="eyJhbGciOi..."
  ACTIVELOOP_API_KEY="eyJhbGciOi..."
  ```

### 2. Organization Name vs. Organization Slug (Crucial!)
Deep Lake identifies datasets using a URI format:
```
hub://<organization_slug_or_username>/<dataset_name>
```

| Field | Right Way ✅ | Wrong Way ❌ | Why? |
|---|---|---|---|
| **Organization / Username** | `sheshpalsingh2024` | `sheshpalsingh2024's Org` | Hub URIs cannot contain spaces, capital letters with apostrophes (`'`), or display names. |
| **Dataset Name** | `langchain_course_from_zero_to_hero` | `My Course Data!` | Slugs must be alphanumeric with underscores or hyphens. |
| **Complete URI** | `hub://sheshpalsingh2024/langchain_course_from_zero_to_hero` | `hub://sheshpalsingh2024's Org/my course` | Activeloop's cloud servers will throw an `InvalidURIError` or `Entity does not exist`. |

> **Tip:** You can always find your exact username/slug in the URL of your Activeloop profile:
> `https://app.activeloop.ai/<your_username_slug>`

### 3. Permissions & Authorization Errors
When Deep Lake tries to create a dataset on the hub:
- If your token belongs to `user_A` and you try to create a dataset under `hub://user_B/...`, you will get:
  ```
  deeplake.AuthorizationError: The dataset doesn't exist and you don't have permissions to create a new one at this location.
  ```
- Make sure the organization/username slug in your `dataset_path` matches the account that issued your token.

---

## 4. Line-by-Line Code Walkthrough

Below is the complete walkthrough of [`03_vectordb.py`](file:///home/sheshpal/Documents/Activloop/langchain_vectordb_in_Production/src/langchain_vectordb_in_production/01-Langchain-zero-to-hero/02-GetStart/03_vectordb.py).

### Step 1: Loading Environment Variables Safely
```python
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load from both the script's local directory and root directory
current_dir = Path(__file__).parent
load_dotenv(current_dir / ".env")
load_dotenv()

# Safely extract the token without crashing if one key is missing
activeloop_token = (
    os.getenv("ACTIVELOOP_TOKEN")
    or os.getenv("ACTIVELOOP_API_KEY")
    or os.getenv("DEEPLAKE_API_KEY")
    or os.getenv("DEEPLAKE_API_TOKEN")
)
if activeloop_token:
    os.environ["ACTIVELOOP_TOKEN"] = activeloop_token
```
- **Why this is needed:** If `os.getenv('KEY')` returns `None`, doing `os.environ["VAR"] = None` raises `TypeError: str expected, not NoneType`. Checking `if activeloop_token:` prevents crashes and supports any naming variation.

---

### Step 2: LangChain 1.x Compatibility Shim
```python
import langchain
import langchain_classic.retrievers
langchain.retrievers = langchain_classic.retrievers
import langchain_classic.retrievers.self_query.base
sys.modules["langchain.retrievers"] = langchain_classic.retrievers
sys.modules["langchain.retrievers.self_query"] = langchain_classic.retrievers.self_query
sys.modules["langchain.retrievers.self_query.base"] = langchain_classic.retrievers.self_query.base
```
- **Why this is needed:** In modern LangChain (v1.x), older modules like `retrievers` were relocated into `langchain-classic`. However, `langchain-deeplake` (v0.1.0) expects `langchain.retrievers` to exist in the main `langchain` package. This small shim bridges the two packages seamlessly.

---

### Step 3: Initializing Models & Splitting Text
```python
from langchain_deeplake import DeeplakeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Models
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. Raw documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638"
]

# 3. Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)
```
- **`RecursiveCharacterTextSplitter`**: Breaks long documents into bite-sized chunks so they fit into embedding models and LLM context windows.
- **`create_documents`**: Converts raw Python strings into LangChain `Document` objects with `page_content` and `metadata`.

---

### Step 4: Connecting to Deep Lake & Storing Vectors
```python
my_activeloop_org_id = "sheshpalsingh2024"
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeeplakeVectorStore(
    dataset_path=dataset_path,
    embedding_function=embeddings,
    token=activeloop_token,
    overwrite=True,
)
db.add_documents(docs)
```
- **`DeeplakeVectorStore`**: The official vector store class for Deep Lake.
- **`embedding_function=embeddings`**: Instructs Deep Lake which embedding model to call when documents or queries are submitted.
- **`token=activeloop_token`**: Authenticates your request to write data to your Activeloop Hub cloud account.
- **`overwrite=True`**: Cleans up previous runs and replaces with the fresh dataset (useful for tutorials/testing).
- **`add_documents(docs)`**: Deep Lake takes each document chunk, calculates its embedding vector using OpenAI, and uploads both the text and vector to the cloud.

---

### Step 5: Building a Retriever & Running RAG
```python
# Create retriever to fetch top 2 most relevant chunks
retriever = db.as_retriever(search_kwargs={"k": 2})

def rag_query(question: str) -> str:
    # 1. Search the vector database for relevant chunks
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # 2. Build prompt with injected context
    prompt = (
        "Use the context below to answer the question. "
        "If you don't know, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    
    # 3. Call the LLM
    return llm.invoke(prompt).content

print(rag_query("When was Napoleon Bonaparte born?"))
```
- When `rag_query("When was Napoleon Bonaparte born?")` is called:
  1. The question is embedded into a vector.
  2. Deep Lake finds that the vector is closest to `"Napoleon Bonaparte was born in 15 August 1769"`.
  3. That text is formatted into the prompt as `Context`.
  4. GPT-4o Mini reads the context and returns:
     > `"Napoleon Bonaparte was born on 15 August 1769."`

---

## 5. Viewing Your Database in the Browser

Whenever your script creates a dataset with `hub://sheshpalsingh2024/langchain_course_from_zero_to_hero`, you can inspect it visually in your web browser:

🔗 **Dataset Direct Link:**
[https://app.activeloop.ai/sheshpalsingh2024/langchain_course_from_zero_to_hero](https://app.activeloop.ai/sheshpalsingh2024/langchain_course_from_zero_to_hero)

### What you see in the Activeloop Web UI:
1. **Columns / Tensors:**
   - `documents`: The text contents of your chunks.
   - `embeddings`: The high-dimensional float arrays (vectors) generated by OpenAI.
   - `ids`: Unique identifiers assigned to each document.
   - `metadata`: Any dictionary tags (source, author, date) attached to chunks.
2. **Visual Table View:** You can scroll through rows, filter by values, and inspect data points.
3. **Storage & Access Settings:** Change visibility between Public and Private, manage access tokens, and check dataset revisions.

---

## 6. Troubleshooting Cheat Sheet

| Error Message | Root Cause | Solution |
|---|---|---|
| `TypeError: str expected, not NoneType` | `os.environ["..."] = os.getenv(...)` returned `None` because the variable was missing from `.env`. | Use `activeloop_token = os.getenv(...)` and only assign `os.environ` if the variable is not `None`. |
| `TypeError: 'module' object is not callable` | Tried to call `db = deeplake(...)` from `from langchain_classic.vectorstores import deeplake`. | Import `from langchain_deeplake import DeeplakeVectorStore` instead. |
| `deeplake.AuthorizationError: The dataset doesn't exist and you don't have permissions` | The organization name in `hub://...` doesn't match the username/org tied to your API token. | Use your exact username slug (e.g., `sheshpalsingh2024`), not display names with spaces or apostrophes. |
| `ModuleNotFoundError: No module named 'langchain.retrievers'` | In LangChain 1.x, retrievers moved to `langchain-classic`. | Add the 4-line `sys.modules` shim shown in Step 2 before importing `langchain_deeplake`. |
| `ModuleNotFoundError: No module named 'dotenv'` | Executing with global system python instead of the project virtual environment. | Run with `./.venv/bin/python <script.py>` or activate your virtual environment (`source .venv/bin/activate`). |

---

## 7. Local vs. Cloud Deep Lake (Offline Mode)

If you don't want to use the cloud Hub or don't have an internet connection, you can store Deep Lake datasets completely offline on your local disk:

```python
# Instead of hub://..., specify a local folder path:
dataset_path = "./data/my_local_deeplake"

db = DeeplakeVectorStore(
    dataset_path=dataset_path,
    embedding_function=embeddings,
    overwrite=True,
)
```
Deep Lake will create the folder locally and store all vectors and tensors right on your machine!
