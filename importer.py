import os
import json
import numpy as np
import pandas as pd
from weaviate.util import generate_uuid5
from tqdm import tqdm
from dataset import wiki, jeopardy


def upload_jeopardy_objects(client):

    client.schema.create_class(jeopardy.class_obj_category)
    client.schema.create_class(jeopardy.class_obj_question)

    # Import data

    question_vec_fname = "./data/jeopardy_10k.json.npy"
    category_vec_fname = "./data/jeopardy_10k_categories.csv"
    data_fname = "./data/jeopardy_10k.json"

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
                if i + 1 <= 10**5:
                    yield (l)


    with client.batch as batch:
        batch.batch_size = 100
        batch.dynamic = False
        for i, data in tqdm(enumerate(load_data(data_fname))):
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

    return True


def upload_winereview_objects(client):

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

    # Add class with description, title, country and points

    client.schema.create_class(class_obj_winereview)

    # Import data

    df = pd.read_csv("./data/winemag_tiny.csv")

    with client.batch as batch:
        for _, row in df.iterrows():
            data_obj = {
                "review_body": row["description"],
                "title": row["title"],
                "country": row["country"],
                "points": row["points"],
                "price": row["price"],
            }
            uuid = generate_uuid5(data_obj)
            batch.add_data_object(data_obj, "WineReview", uuid=uuid) 

    return True


# TODO - refactoring

# Data uploader should receive data from a loader: row by row (needs: class name, vectorizer, a set of original column: wv column & type mappings, possibly the vector)
    # Create schema
    # Set batch props
    # Add to a batch row by row
    # For Xrefs - use UUID generator for object
