import os
from deeplake import Client

client = Client(
    token=os.environ.get("DEEPLAKE_API_KEY"),
    workspace_id=os.environ.get("DEEPLAKE_WORKSPACE"),
)