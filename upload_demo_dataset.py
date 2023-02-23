import weaviate
from weaviate.util import generate_uuid5
import json
import os
import numpy as np
import pandas as pd

# ===== INSTANTIATE CLIENT =====

wv_url = "https://academytest.weaviate.network"
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

print("Client instantiated")

# ===== Set up batch params =====

client.batch.batch_size = 50
client.batch.dynamic = True

# ===== POPULATE WIKI CLASS =====

class_obj_wiki = {
    "class": "WikiArticle",
    "description": "A Wikipedia article",
    "properties": [
        {"name": "title", "dataType": ["text"]},
        {"name": "url", "dataType": ["string"]},
        {"name": "wiki_summary", "dataType": ["text"]},
    ],
    "vectorizer": "text2vec-openai",
}

client.schema.create_class(class_obj_wiki)

# Import data

for dfile in [
    f for f in os.listdir("./data") if f.startswith("wiki") and f.endswith(".json")
]:
    with open(os.path.join("./data", dfile), "r") as f:
        data = json.load(f)
    data_obj = {
        "title": data["title"],
        "url": data["url"],
        "wiki_summary": data["summary"],
    }

    uuid = generate_uuid5(data_obj)

    with client.batch() as batch:
        batch.add_data_object(data_obj, "WikiArticle", uuid=uuid)

print("Added Wiki objects")

# ===== POPULATE WineReview CLASS =====

# Add class with description, title, country and points

class_obj_winereview = {
    "class": "WineReview",
    "vectorizer": "text2vec-openai",
    "properties": [
        {"name": "review_body", "dataType": ["text"], "description": "Review body"},
        {"name": "title", "dataType": ["string"], "description": "Name of the wine"},
        {
            "name": "country",
            "dataType": ["string"],
            "description": "Originating country",
        },
        {
            "name": "points",
            "dataType": ["int"],
            "description": "Review score in points",
        },
        {"name": "price", "dataType": ["number"], "description": "Listed price"},
    ],
}

client.schema.create_class(class_obj_winereview)

# Import data

df = pd.read_csv("./data/winemag_tiny.csv")

with client.batch as batch:
    for i, row in df.iterrows():
        data_obj = {
            "review_body": row["description"],
            "title": row["title"],
            "country": row["country"],
            "points": row["points"],
            "price": row["price"],
        }
        uuid = generate_uuid5(data_obj)
        batch.add_data_object(data_obj, "WineReview", uuid=uuid)

print("Added WineReview objects")

# ===== POPULATE Jeopardy CLASSes =====

class_obj_category = {
    "class": "JeopardyCategory",
    "description": "A Jeopardy! category",
    "vectorizer": "text2vec-openai",
}

class_obj_question = {
    "class": "JeopardyQuestion",
    "description": "A Jeopardy! question",
    "vectorizer": "text2vec-openai",
    "properties": [
        {
            "name": "hasCategory",
            "dataType": ["JeopardyCategory"],
            "description": "The category of the question",
        },
        {
            "name": "question",
            "dataType": ["text"],
            "description": "Question asked to the contestant",
        },
        {
            "name": "answer",
            "dataType": ["text"],
            "description": "Answer provided by the contestant",
        },
        {
            "name": "value",
            "dataType": ["int"],
            "description": "Points that the question was worth",
        },
        {
            "name": "round",
            "dataType": ["string"],
            "description": "Jeopardy round",
        },
        {
            "name": "air_date",
            "dataType": ["date"],
            "description": "Date that the episode first aired on TV",
        },
    ],
}

client.schema.create_class(class_obj_category)
client.schema.create_class(class_obj_question)

# Import data

question_vec_fname = "./data/jeopardy_1k.json.npy"
category_vec_fname = "./data/jeopardy_1k_categories.csv"
data_fname = "./data/jeopardy_1k.json"

vec_array = np.load(question_vec_fname)
cat_df = pd.read_csv(category_vec_fname)
cat_arr = cat_df.iloc[:, :-1].to_numpy()
cat_names = cat_df["category"].to_list()
cat_emb_dict = dict(zip(cat_names, cat_arr))


def get_question_obj(data: dict) -> dict:
    from datetime import datetime, timezone

    question_obj = {
        "question": data["Question"],
        "answer": data["Answer"],
        "value": data["Value"],
        "round": data["Round"],
        "air_date": datetime.strptime(data["Air Date"], "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .isoformat(),
    }
    return question_obj


def get_category_obj(data: dict) -> dict:
    category_obj = {"title": data["Category"]}
    return category_obj


def load_data(fpath):
    with open(fpath, "r") as f:
        data = json.load(f)
        for i, l in enumerate(data):
            if i + 1 <= 100:
                yield (l)


with client.batch as batch:
    for i, data in enumerate(load_data(data_fname)):
        # Create question object, uuid & emb
        question_obj = get_question_obj(data)
        question_id = generate_uuid5(question_obj)

        # Create category object & uuid
        category_obj = get_category_obj(data)
        category_id = generate_uuid5(category_obj)

        # Get embeddings
        question_emb = vec_array[i].tolist()
        category_emb = list(cat_emb_dict[data["Category"]])

        # Add data objects to the batch
        batch.add_data_object(
            question_obj, "JeopardyQuestion", uuid=question_id, vector=question_emb
        )
        batch.add_data_object(
            category_obj, "JeopardyCategory", uuid=category_id, vector=category_emb
        )

        # Add reference from question to category
        batch.add_reference(
            from_object_uuid=question_id,
            from_object_class_name="JeopardyQuestion",
            from_property_name="hasCategory",
            to_object_uuid=category_id,
            to_object_class_name="JeopardyCategory",
        )

print("Added Jeopardy objects")
