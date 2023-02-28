import weaviate
from weaviate.util import generate_uuid5
import os
import importer
from dataset import wiki

# ===== INSTANTIATE CLIENT =====

# wv_url = "https://academytest.weaviate.network"
wv_url = "https://jptest.weaviate.network"
api_key = os.environ.get("OPENAI_API_KEY")

auth = weaviate.AuthClientPassword(
    username=os.environ.get("WCS_USER"),
    password=os.environ.get("WCS_PASS"),
)

client = weaviate.Client(
    url=wv_url,
    auth_client_secret=auth,
    additional_headers={"X-OpenAI-Api-Key": api_key},
)

client.schema.delete_all()
print("Client instantiated")

# ===== Set up batch params =====

wiki.upload_objects(client, batch_size=1)
print("Uploaded wiki objects")
# importer.upload_jeopardy_objects(client)  
# print("Uploaded Jeopardy objects")
# importer.upload_winereview_objects(client)
# print("Uploaded WineReview")
