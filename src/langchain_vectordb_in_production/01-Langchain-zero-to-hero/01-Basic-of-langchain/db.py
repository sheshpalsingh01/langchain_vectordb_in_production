import os
import deeplake
from dotenv import load_dotenv
load_dotenv()

from langchain_classic.vectorstores import deeplake
from langchain_ollama import OllamaEmbeddings


#====================================
# Embeeding model
#====================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:v1.5"
)

vector = embeddings.embed_query(
    "What is machine learning?"
)
print("Embedding dimensions:", len(vector))


#======================================================
# Local Deeplake
#======================================================
import deeplake

print("Deep Lake version:", deeplake.__version__)

db = deeplake.create("./data/deeplake1")

print("Deep Lake created successfully")

#=====================================================
# cloud Deeplake
#=====================================================
print(dir(deeplake.Client))

token = os.getenv("ACTIVELOOP_TOKEN")

db = deeplake.create(
    
)

print(db)


#==========================================
#
#==========================================


token = os.environ["ACTIVELOOP_TOKEN"]
org_id = os.environ["DEEPLAKE_ORG_ID"]

dataset_path = f"al://{org_id}/rag-documents"

db = deeplake.create(
    dataset_path,
    token=token,
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:v1.5"
)

print("Connected to:", dataset_path)

#==================================#
# 

import os
from deeplake import Client
from dotenv import load_dotenv

load_dotenv()

client = Client(
    token=os.environ["DEEOLAKE_API_TOKEN"],
    workspace_id=os.environ["DEEPLAKE_WORKSPACE"],
)
# 1. See existing tables
print(client.list_tables())

# 2. Create your vector-store table
client.create_table(schema =)

# 3. Open it
table = client.open_table(...)

# 4. Insert documents + Nomic embeddings
# table / client ingest operations

# 5. Create vector index
client.create_index(...)

# 6. Search
client.query(...)