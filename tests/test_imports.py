import weaviate
import weaviate_datasets as wd
import os

client = weaviate.connect_to_local(
    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY")
    }
)

# Just a temporary test script to manually check if the datasets are uploaded correctly
for dataset in [wd.JeopardyQuestions1k, wd.JeopardyQuestions10k]:
    d = dataset()
    d.upload_dataset(client, overwrite=True, compress=True)
    c = client.collections.get("JeopardyQuestion")
    print(len(c))  # Should be 1000 and 10000

for dataset in [wd.WineReviewsNV]:
    d = dataset()
    d.upload_dataset(client, overwrite=True, compress=True)
    c = client.collections.get(d.collection_name)
    print(len(c))  # Should be 50

client.close()
