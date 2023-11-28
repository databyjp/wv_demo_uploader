import os
import pandas as pd
from weaviate.util import generate_uuid5
from weaviate import WeaviateClient
from weaviate.classes import Configure, Property, ReferenceProperty, ReferencePropertyMultiTarget, DataType, Tokenization
from weaviate.collections.collection import Collection
from tqdm import tqdm
import numpy as np
import json
import logging


logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

basedir = os.path.dirname(os.path.abspath(__file__))


class WineReviews:

    collection_name = "WineReview"
    winedata_path = os.path.join(basedir, "data", "winemag_tiny.csv")

    def add_to_schema(self, client: WeaviateClient) -> Collection:
        """
        For each class in the dataset, add its definition to the Weaviate instance.
        """
        reviews = client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            generative_config=Configure.Generative.openai(),
            properties=[
                Property(
                    name="review_body",
                    data_type=DataType.TEXT,
                    description="Review body"
                ),
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    description="Name of the wine"
                ),
                Property(
                    name="country",
                    data_type=DataType.TEXT,
                    description="Originating country"
                ),
                Property(
                    name="points",
                    data_type=DataType.INT,
                    description="Review score in points"
                ),
                Property(
                    name="price",
                    data_type=DataType.NUMBER,
                    description="Listed price"
                ),
            ]
        )
        return reviews

    def _class_dataloader(self):
        df = pd.read_csv(self.winedata_path)
        for _, row in df.iterrows():
            data_obj = {
                "review_body": row["description"],
                "title": row["title"],
                "country": row["country"],
                "points": row["points"],
                "price": row["price"],
            }
            yield data_obj, None

    def upload_objects(
        self, client: WeaviateClient, batch_size=200
    ) -> bool:
        """
        Base uploader method for uploading a single class.
        """
        client.batch.configure(batch_size=batch_size)
        with client.batch as batch:
            for data_obj, vector in tqdm(self._class_dataloader()):
                uuid = generate_uuid5(data_obj)
                batch.add_object(
                    properties=data_obj,
                    collection=self.collection_name,
                    uuid=uuid,
                    vector=vector,
                )

        return True

    def upload_dataset(self, client: WeaviateClient, batch_size=300, overwrite=False) -> bool:
        """
        Adds the class to the schema,
        then calls `upload_objects` to upload the objects.
        """
        if overwrite:
            client.collections.delete(self.collection_name)
        _ = self.add_to_schema(client)
        _ = self.upload_objects(client, batch_size=batch_size)
        return True


class JeopardyQuestions1k:
    data_fpath = os.path.join(basedir, "data", "jeopardy_1k.json")
    arr_fpath = os.path.join(basedir, "data", "jeopardy_1k.json.npy")
    category_vec_fpath = os.path.join(basedir, "data", "jeopardy_1k_categories.csv")

    question_collection = "JeopardyQuestion"
    category_collection = "JeopardyCategory"
    xref_prop_name = "hasCategory"

    def add_to_schema(self, client: WeaviateClient) -> Collection:
        """
        For each class in the dataset, add its definition to the Weaviate instance.
        """
        categories = client.collections.create(
            name=self.category_collection,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            generative_config=Configure.Generative.openai(),
            properties=[
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    description="The category title"
                )
            ]
        )

        questions = client.collections.create(
            name=self.question_collection,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            generative_config=Configure.Generative.openai(),
            inverted_index_config=Configure.inverted_index(
                index_property_length=True,
                index_timestamps=True,
                index_null_state=True
            ),
            properties=[
                ReferenceProperty(
                    name=self.xref_prop_name,
                    target_collection="JeopardyQuestion",
                ),
                Property(
                    name="question",
                    data_type=DataType.TEXT,
                    description="Question asked to the contestant"
                ),
                Property(
                    name="answer",
                    data_type=DataType.TEXT,
                    description="Answer provided by the contestant"
                ),
                Property(
                    name="points",
                    data_type=DataType.INT,
                    description="Jeopardy points"
                ),
                Property(
                    name="round",
                    data_type=DataType.TEXT,
                    description="Jeopardy round",
                    tokenization=Tokenization.FIELD
                ),
                Property(
                    name="air_date",
                    data_type=DataType.DATE,
                    description="Date that the episode first aired on TV"
                ),
            ]
        )
        return categories, questions

    def _class_pair_dataloader(self):
        from datetime import datetime, timezone

        question_vec_array = np.load(self.arr_fpath)
        category_vec_dict = self._get_cat_array()

        with open(self.data_fpath, "r") as f:
            data = json.load(f)
            for i, row in enumerate(data):
                try:
                    question_obj = {
                        "question": row["Question"],
                        "answer": row["Answer"],
                        "points": row["Value"],
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

    def upload_objects(
        self, client: WeaviateClient, batch_size=200
    ) -> bool:
        """
        Base uploader method for uploading a single class.
        """
        client.batch.configure(batch_size=batch_size)
        with client.batch as batch:
            for (data_obj_from, vec_from), (data_obj_to, vec_to) in tqdm(
                self._class_pair_dataloader()
            ):
                # Add "class_from" objects
                id_from = generate_uuid5(data_obj_from)
                batch.add_object(
                    properties=data_obj_from,
                    collection=self.question_collection,
                    uuid=id_from,
                    vector=vec_from,
                )

                # Add "class_to" objects
                id_to = generate_uuid5(data_obj_to)
                batch.add_object(
                    properties=data_obj_to,
                    collection=self.category_collection,
                    uuid=id_to,
                    vector=vec_to,
                )

                # Add references
                batch.add_reference(
                    from_object_collection=self.question_collection,
                    from_object_uuid=id_from,
                    to_object_collection=self.category_collection,
                    to_object_uuid=id_to,
                    from_property_name=self.xref_prop_name
                )

        return True

    def upload_dataset(self, client: WeaviateClient, batch_size=300, overwrite=False) -> bool:
        """
        Adds the class to the schema,
        then calls `upload_objects` to upload the objects.
        """
        if overwrite:
            client.collections.delete(self.question_collection)
            client.collections.delete(self.category_collection)

        _ = self.add_to_schema(client)
        _ = self.upload_objects(client, batch_size=batch_size)
        return True
