from langchain_deeplake.vectorstores import DeeplakeVectorStore
from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(
    model="nomic-embed-text:v1.5"
)


db = DeeplakeVectorStore(
    dataset_path="./data/deeplake",
    embedding_function=embeddings,
)


import deeplake

print("Deep Lake version:", deeplake.__version__)

db = deeplake.create("./data/deeplake")

print("Deep Lake created successfully")