import os
import json
import pandas as pd
import numpy as np
from weaviate.util import generate_uuid5
from weaviate import Client
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

basedir = os.path.dirname(os.path.abspath(__file__))


class Dataset:

    """
    TODO: 
    Add .dataset_size method to get dataset size
    Add tests!
    """

    def __init__(self):
        self._class_definitions = []

    def get_class_definitions(self) -> list:
        return self._class_definitions

    def get_class_names(self) -> list:
        return [c["class"] for c in self._class_definitions]

    # ----- METHODS FOR SCHEMA DETAILS -----

    def _class_in_schema(self, client: Client, class_name) -> bool:
        schema = client.schema.get()
        return class_name in [wv_class["class"] for wv_class in schema["classes"]]

    def classes_in_schema(self, client: Client) -> dict:
        """
        Polls the Weaviate instance to check if this class exists.
        """
        class_names = self.get_class_names()
        return {
            class_name: self._class_in_schema(client, class_name)
            for class_name in class_names
        }
    
    def delete_existing_dataset_classes(self, client: Client) -> bool:
        class_names = self.get_class_names()
        for class_name in class_names:
            if self._class_in_schema(client, class_name):
                client.schema.delete_class(class_name)
        return True

    def add_to_schema(self, client: Client) -> bool:
        results = dict()
        for wv_class in self._class_definitions:
            class_name = wv_class["class"]
            if not self._class_in_schema(client, class_name):
                client.schema.create_class(wv_class)
                status = f"{class_name}: {self._class_in_schema(client, class_name)}"
                results[class_name] = status
            else:
                results[class_name] = "Already present; not added."
        return results

    def set_vectorizer(self, vectorizer_name: str, module_config: dict) -> list:
        # TODO - add vectorizer / module config validation here
        for i, class_def in enumerate(self._class_definitions):
            class_def["vectorizer"] = vectorizer_name
            class_def["moduleConfig"] = module_config
        return self.get_class_definitions()

    # ----- DATA UPLOADER GENERIC METHODS -----

    def _class_uploader(
        self, client: Client, class_name: str, batch_size: int = 100
    ) -> bool:
        with client.batch() as batch:
            batch.batch_size = batch_size
            for data_obj, vector in tqdm(self._class_dataloader(class_name)):
                uuid = generate_uuid5(data_obj)
                batch.add_data_object(data_obj, class_name, uuid=uuid, vector=vector)

        return True

    def _class_pair_uploader(
        self, client: Client, class_from: str, class_to: str, batch_size: int = 100
    ) -> bool:
        with client.batch() as batch:
            batch.batch_size = batch_size
            for (data_obj_from, vec_from), (data_obj_to, vec_to) in tqdm(
                self._class_pair_dataloader()
            ):
                # Add "class_from" objects
                id_from = generate_uuid5(data_obj_from)
                batch.add_data_object(
                    data_obj_from,
                    class_from,
                    uuid=id_from,
                    vector=vec_from,
                )
                # Add "class_to" objects
                id_to = generate_uuid5(data_obj_to)
                batch.add_data_object(
                    data_obj_to,
                    class_to,
                    uuid=id_to,
                    vector=vec_to,
                )

                # Add references
                class_def = [
                    c for c in self._class_definitions if c["class"] == class_from
                ][0]
                xref_props = [
                    p for p in class_def["properties"] if p["dataType"][0] == class_to
                ]
                if len(xref_props) > 0:
                    xref_prop_def = xref_props[0]
                    batch.add_reference(
                        from_object_uuid=id_from,
                        from_object_class_name=class_from,
                        from_property_name=xref_prop_def["name"],
                        to_object_uuid=id_to,
                        to_object_class_name=class_to,
                    )

        return True

    def upload_dataset(self, client: Client, batch_size=100) -> bool:
        """
        Adds the class to the schema, then
        Upload the objects.
        """
        schema_add_results = self.add_to_schema(client)
        self.upload_objects(client, batch_size=batch_size)
        return True


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

    def _class_dataloader(self, class_name):
        if class_name == "WikiArticle":
            datasets_dir = os.path.join(basedir, "data")
            for dfile in [
                f
                for f in os.listdir(datasets_dir)
                if f.startswith("wiki") and f.endswith(".json")
            ]:
                with open(os.path.join(datasets_dir, dfile), "r") as f:
                    data = json.load(f)

                data_obj = {
                    "title": data["title"],
                    "url": data["url"],
                    "wiki_summary": data["summary"],
                }
                yield data_obj, None
        else:
            raise ValueError("Unknown class name")

    def upload_objects(self, client: Client, batch_size: int) -> bool:
        for class_name in self.get_class_names():
            self._class_uploader(client, class_name, batch_size)
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

    def _class_dataloader(self, class_name):
        winedata_path = os.path.join(basedir, "data", "winemag_tiny.csv")
        if class_name == "WineReview":
            df = pd.read_csv(winedata_path)
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

    def upload_objects(self, client: Client, batch_size: int) -> bool:
        for class_name in self.get_class_names():
            self._class_uploader(client, class_name, batch_size)
        return True


class JeopardyQuestions1k(Dataset):

    data_fpath = os.path.join(basedir, "data", "jeopardy_1k.json")
    arr_fpath = os.path.join(basedir, "data", "jeopardy_1k.json.npy")
    category_vec_fpath = os.path.join(basedir, "data", "jeopardy_1k_categories.csv")

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

    def _class_pair_dataloader(self, test_mode=False):
        from datetime import datetime, timezone

        if test_mode:  # Added to this function for testing as data size non trivial
            max_objs = 150
        else:
            max_objs = 10**10

        question_vec_array = np.load(self.arr_fpath)
        category_vec_dict = self._get_cat_array()

        with open(self.data_fpath, "r") as f:
            data = json.load(f)
            for i, row in enumerate(data):
                try:
                    if i >= max_objs:
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
                except:
                    logging.warning(f"Data parsing error on row {i}")

    def _get_cat_array(self) -> dict:
        cat_df = pd.read_csv(self.category_vec_fpath)
        cat_arr = cat_df.iloc[:, :-1].to_numpy()
        cat_names = cat_df["category"].to_list()
        cat_emb_dict = dict(zip(cat_names, cat_arr))
        return cat_emb_dict

    def upload_objects(self, client: Client, batch_size: int) -> bool:
        return self._class_pair_uploader(
            client,
            class_from="JeopardyQuestion",
            class_to="JeopardyCategory",
            batch_size=batch_size,
        )


class JeopardyQuestions10k(JeopardyQuestions1k):

    data_fpath = os.path.join(basedir, "data", "jeopardy_10k.json")
    arr_fpath = os.path.join(basedir, "data", "jeopardy_10k.json.npy")
    category_vec_fpath = os.path.join(basedir, "data", "jeopardy_10k_categories.csv")