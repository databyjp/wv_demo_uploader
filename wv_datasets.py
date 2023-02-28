import os
import json
import pandas as pd
import numpy as np
from weaviate.util import generate_uuid5
from weaviate import Client
from tqdm import tqdm


class Dataset:
    def __init__(self):
        self._class_definitions = []

    def _obj_loader(self):
        for data_obj in [{}]:
            yield data_obj

    def see_class_definitions(self):
        return self._class_definitions

    def get_class_names(self):
        return [c["class"] for c in self._class_definitions]

    def _class_in_schema(self, client: Client, class_name):
        schema = client.schema.get()
        return class_name in [wv_class["class"] for wv_class in schema["classes"]]

    def classes_in_schema(self, client: Client):
        """
        Polls the Weaviate instance to check if this class exists.
        """
        class_names = self.get_class_names()
        return {
            class_name: self._class_in_schema(client, class_name)
            for class_name in class_names
        }

    def add_to_schema(self, client: Client) -> bool:
        results = dict()
        for wv_class in self._class_definitions:
            class_name = wv_class["class"]
            if not self._class_in_schema(client, class_name):
                client.schema.create_class(wv_class)
                status = f"{class_name}: {self._class_in_schema(client, class_name)}"
                results[class_name] = status
            else:
                results[class_name] = "Already present"
        return results


class WikiArticles(Dataset):
    def __init__(self):
        self._class_definitions = [
            {
                "class": "WikiArticle",
                "description": "A Wikipedia article",
                "properties": [
                    {"name": "title", "dataType": ["text"]},
                    {"name": "url", "dataType": ["string"]},
                    {"name": "wiki_summary", "dataType": ["text"]},
                ],
                "vectorizer": "text2vec-openai",
                "moduleConfig": {
                    "qna-openai": {
                        "model": "text-davinci-002",
                        "maxTokens": 16,
                        "temperature": 0.0,
                        "topP": 1,
                        "frequencyPenalty": 0.0,
                        "presencePenalty": 0.0,
                    }
                },
            }
        ]

    def _obj_loader(self, class_name):
        if class_name == "WikiArticle":
            for dfile in [
                f
                for f in os.listdir("./data")
                if f.startswith("wiki") and f.endswith(".json")
            ]:
                with open(os.path.join("./data", dfile), "r") as f:
                    data = json.load(f)

                data_obj = {
                    "title": data["title"],
                    "url": data["url"],
                    "wiki_summary": data["summary"],
                }
                yield data_obj, None
        else:
            raise ValueError("Unknown class name")            

    def upload_objects(self, client: Client, batch_size=50) -> bool:
        with client.batch() as batch:
            batch.batch_size = batch_size
            for class_name in self.get_class_names():
                for data_obj, vector in self._obj_loader(class_name):
                    uuid = generate_uuid5(data_obj)
                    batch.add_data_object(data_obj, class_name, uuid=uuid, vector=vector)

        return True


class WineReviews(Dataset):
    def __init__(self):
        self._class_definitions = [
            {
                "class": "WineReview",
                "vectorizer": "text2vec-openai",
                "properties": [
                    {
                        "name": "review_body",
                        "dataType": ["text"],
                        "description": "Review body",
                    },
                    {
                        "name": "title",
                        "dataType": ["string"],
                        "description": "Name of the wine",
                    },
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
                    {
                        "name": "price",
                        "dataType": ["number"],
                        "description": "Listed price",
                    },
                ],
            }
        ]

    def _obj_loader(self, class_name):
        if class_name == "WineReview":
            df = pd.read_csv("./data/winemag_tiny.csv")
            for _, row in df.iterrows():
                data_obj = {
                    "review_body": row["description"],
                    "title": row["title"],
                    "country": row["country"],
                    "points": row["points"],
                    "price": row["price"],
                }
                yield data_obj, None
        else:
            raise ValueError("Unknown class name")

    def upload_objects(self, client: Client, batch_size=50) -> bool:
        with client.batch() as batch:
            batch.batch_size = batch_size
            for class_name in self.get_class_names():
                for data_obj, vector in self._obj_loader(class_name):
                    uuid = generate_uuid5(data_obj)
                    batch.add_data_object(data_obj, class_name, uuid=uuid, vector=vector)

        return True


class JeopardyQuestions(Dataset):
    def __init__(self):
        self._class_definitions = [
            {
                "class": "JeopardyCategory",
                "description": "A Jeopardy! category",
                "vectorizer": "text2vec-openai",
            },
            {
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
            },
        ]

    def _obj_loader(self):
        from datetime import datetime, timezone

        data_fname = "./data/jeopardy_10k.json"
        question_vec_array = np.load("./data/jeopardy_10k.json.npy")
        category_vec_dict = self._get_cat_array()

        with open(data_fname, "r") as f:
            data = json.load(f)
            for i, row in enumerate(data):
                if i >= 100:
                    break
                else:
                    question_obj = {
                        "question": row["Question"],
                        "answer": row["Answer"],
                        "value": row["Value"],
                        "round": row["Round"],
                        "air_date": datetime.strptime(row["Air Date"], "%Y-%m-%d")
                        .replace(tzinfo=timezone.utc)
                        .isoformat(),
                    }
                    question_vec = question_vec_array[i].tolist()
                    category_obj = {"title": row["Category"]}
                    category_vec = list(category_vec_dict[category_obj["title"]])
                    yield (question_obj, question_vec), (category_obj, category_vec)

    def _get_cat_array(self):
        category_vec_fname = "./data/jeopardy_10k_categories.csv"
        cat_df = pd.read_csv(category_vec_fname)
        cat_arr = cat_df.iloc[:, :-1].to_numpy()
        cat_names = cat_df["category"].to_list()
        cat_emb_dict = dict(zip(cat_names, cat_arr))
        return cat_emb_dict

    def upload_objects(self, client: Client, batch_size=30) -> bool:

        with client.batch() as batch:
            batch.batch_size = batch_size
            for (question_obj, question_vec), (category_obj, category_vec) in tqdm(self._obj_loader()):  # TODO - refactor this to take two arguments for class names
                # Add Question object including embedding
                question_id = generate_uuid5(question_obj)
                batch.add_data_object(
                    question_obj,
                    "JeopardyQuestion",
                    uuid=question_id,
                    vector=question_vec,
                )
                # Add Category object including embedding
                category_id = generate_uuid5(category_obj)
                batch.add_data_object(
                    category_obj,
                    "JeopardyCategory",
                    uuid=category_id,
                    vector=category_vec,
                )
                # Add reference from question to category
                batch.add_reference(
                    from_object_uuid=question_id,
                    from_object_class_name="JeopardyQuestion",
                    from_property_name="hasCategory",
                    to_object_uuid=category_id,
                    to_object_class_name="JeopardyCategory",
                )









            # for class_name in self.get_class_names():
            #     for i, (question_obj, category_obj) in tqdm(
            #         enumerate(self._obj_loader(class_name))
            #     ):
            #         # Add Question object including embedding
            #         question_emb = question_vec_array[i].tolist()
            #         question_id = generate_uuid5(question_obj)
            #         batch.add_data_object(
            #             question_obj,
            #             "JeopardyQuestion",
            #             uuid=question_id,
            #             vector=question_emb,
            #         )
            #         # Add Category object including embedding
            #         category_emb = list(category_vec_dict[category_obj["title"]])
            #         category_id = generate_uuid5(category_obj)
            #         batch.add_data_object(
            #             category_obj,
            #             "JeopardyCategory",
            #             uuid=category_id,
            #             vector=category_emb,
            #         )
            #         # Add reference from question to category
            #         batch.add_reference(
            #             from_object_uuid=question_id,
            #             from_object_class_name="JeopardyQuestion",
            #             from_property_name="hasCategory",
            #             to_object_uuid=category_id,
            #             to_object_class_name="JeopardyCategory",
            #         )

            # for class_name in self.get_class_names():
            #     for i, (data_obj, vec) in tqdm(
            #         enumerate(self._obj_loader(class_name))
            #     ):
            #         uuid = generate_uuid5(data_obj)
            #         batch.add_data_object(
            #             data_obj,
            #             class_name,
            #             uuid=uuid,
            #             vector=vec,
            #         )

            # for class_name in self.get_class_names():
            #     class_def = [c for c in self._class_definitions if c["class"] == class_name][0]
            #     xref_props = [p for p in class_def["properties"] if p["dataType"] in self.get_class_names()]
            #     for xref_prop in xref_props:
            #         xref_prop_def = [prop_def for prop_def in class_def["properties"] if prop_def["dataType"][0] == xref_prop]
            #         for i, (data_obj, vec) in tqdm(
            #             enumerate(self._obj_loader(class_name))
            #         ):                    
            #         batch.add_reference(
            #             from_object_uuid=question_id,
            #             from_object_class_name=class_name,
            #             from_property_name=xref_prop_def["name"],
            #             to_object_uuid=category_id,
            #             to_object_class_name=xref_prop,
            #         )
        return True
