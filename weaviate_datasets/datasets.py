import os
from typing import Dict, Tuple, Union, List, Generator, Literal
from pathlib import Path
import pandas as pd
from weaviate.util import generate_uuid5
from weaviate import WeaviateClient, Client
from weaviate.classes import (
    Configure,
    Property,
    ReferenceProperty,
    DataType,
    Tokenization,
    Tenant,
    DataObject,
)
from weaviate.collections.collection import Collection
from tqdm import tqdm
import numpy as np
import json
import logging
import time


logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

basedir = os.path.dirname(os.path.abspath(__file__))


def wiki_parser(
    wiki_text: str, heading_only: bool = False, chunk_sections: bool = False
) -> List[Dict[str, str]]:
    lines = wiki_text.split("\n")
    sections = []
    current_section = {"heading": "", "body": ""}
    current_heading = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("*"):
            # Update current section if it has content
            if current_section["body"]:
                sections.append(current_section)
                current_section = {"heading": "", "body": ""}

            # Update heading levels
            depth = line[:5].split(":")[0].count("*")
            heading_loc = line.find(" -")
            if heading_loc != -1:
                line_heading = line[: line.find(" -")]
            else:
                line_heading = " NO HEADING FOUND "
            current_heading = current_heading[: depth - 1] + [line_heading]
            current_section["heading"] = " | ".join(current_heading)
            current_section["body"] = line[line.find(" -") :]
        else:
            # Continuation of the previous heading's body
            current_section["body"] += line + "\n"

    # Add the last section if it has content
    if current_section["body"]:
        sections.append(current_section)

    # Handle secction dictionaries
    if heading_only:
        return [s["heading"] for s in sections]
    else:
        sections = [s for s in sections if len(s["body"]) > 30]
        if chunk_sections:
            chunks = list()
            for s in sections:
                section_chunks = chunk_string(s["body"])
                for c in section_chunks:
                    chunks.append(s["heading"] + " | " + c)
            return chunks
        else:
            return [s["heading"] + " | " + s["body"] for s in sections]


def chunk_string(s, chunk_size=200, overlap=20):
    chunks = []
    start = 0

    while start < len(s):
        end = start + chunk_size
        end = min(end, len(s))

        chunks.append(s[start:end])

        # Move start for the next chunk
        start = end

        # Apply overlap from the second chunk onwards
        if start < len(s):
            start -= overlap

    return chunks


class SimpleDataset:
    collection_name = None
    vectorizer_config = Configure.Vectorizer.text2vec_openai()
    generative_config = Configure.Generative.openai()
    mt_config = None
    tenants = []
    properties = list()

    def add_collection(self, client: WeaviateClient) -> Collection:
        """
        For each class in the dataset, add its definition to the Weaviate instance.
        """
        collection = client.collections.create(
            name=self.collection_name,
            vectorizer_config=self.vectorizer_config,
            generative_config=self.generative_config,
            properties=self.properties,
            multi_tenancy_config=self.mt_config,
        )
        if self.mt_config is not None:
            collection.tenants.create(self.tenants)

        return collection

    def _class_dataloader(self) -> Generator:
        yield {}, None

    def upload_objects(self, client: WeaviateClient, batch_size=200) -> List:
        """
        Base uploader method for uploading a single class.
        """

        def batch_insert_many(tgt_collection: Collection):
            manual_batch = list()
            responses = list()
            counter = 0
            for data_obj, vector in tqdm(self._class_dataloader()):
                obj = DataObject(
                    properties=data_obj,
                    uuid=generate_uuid5(data_obj),
                    vector=vector
                )
                manual_batch.append(obj)
                counter += 1
                if counter % batch_size == 0:
                    resp = tgt_collection.data.insert_many(manual_batch)
                    responses.append(resp)
                    manual_batch = list()
                    time.sleep(1)  # TODO - fix this stupid time limit / turn it into a parameter
            resp = tgt_collection.data.insert_many(manual_batch)
            responses.append(resp)
            return responses


        collection = client.collections.get(self.collection_name)
        if self.mt_config is None:
            responses = batch_insert_many(collection)
        else:
            for tenant in self.tenants:
                tenant_collection = collection.with_tenant(tenant.name)
                responses = batch_insert_many(tenant_collection)
        return responses


    def upload_dataset(
        self, client: WeaviateClient, batch_size=200, overwrite=False
    ) -> List:
        """
        Adds the class to the schema,
        then calls `upload_objects` to upload the objects.
        """
        if len(self.tenants) == 0 and self.mt_config is not None:
            raise ValueError(
                "A list of tenants is required with multi-tenancy switched on."
            )

        if type(client) == Client:
            raise TypeError(
                "Sorry, this is for the `v4` Weaviate Python Client, with the WeaviateClient object type. Please refer to the README for more information."
            )

        if overwrite:
            client.collections.delete(self.collection_name)

        _ = self.add_collection(client)
        upload_responses = self.upload_objects(client, batch_size=batch_size)

        return upload_responses

    def get_sample(self) -> Dict:
        dl = self._class_dataloader()
        data_obj, _ = next(dl)

        return data_obj


class WineReviews(SimpleDataset):
    def __init__(self):
        self.collection_name = "WineReview"
        self.winedata_path = os.path.join(basedir, "data", "winemag_tiny.csv")
        self.vectorizer_config = Configure.Vectorizer.text2vec_openai()
        self.generative_config = Configure.Generative.openai()
        self.properties = [
            Property(
                name="review_body", data_type=DataType.TEXT, description="Review body"
            ),
            Property(
                name="title", data_type=DataType.TEXT, description="Name of the wine"
            ),
            Property(
                name="country",
                data_type=DataType.TEXT,
                description="Originating country",
            ),
            Property(
                name="points",
                data_type=DataType.INT,
                description="Review score in points",
            ),
            Property(
                name="price", data_type=DataType.NUMBER, description="Listed price"
            ),
        ]

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


class WineReviewsMT(WineReviews):
    def __init__(self):
        super().__init__()
        self.collection_name = "WineReviewMT"
        self.mt_config = Configure.multi_tenancy(enabled=True)
        self.tenants = [Tenant(name="tenantA"), Tenant(name="tenantB")]


class Wiki100(SimpleDataset):
    def __init__(self):
        self.collection_name = "WikiChunk"
        self.article_dir = Path(basedir) / "data/wiki100"
        self.vectorizer_config = Configure.Vectorizer.text2vec_openai()
        self.generative_config = Configure.Generative.openai()
        self.properties = [
            Property(
                name="title", data_type=DataType.TEXT, description="Article title"
            ),
            Property(name="chunk", data_type=DataType.TEXT, description="Text chunk"),
            Property(
                name="chunk_number",
                data_type=DataType.INT,
                description="Chunk number - 1 index",
            ),
        ]
        self.chunking = "wiki_sections"

    def set_chunking(
        self,
        chunking_method: Literal[
            "fixed", "wiki_sections", "wiki_sections_chunked", "wiki_heading_only"
        ],
    ):
        self.chunking = chunking_method

    def _class_dataloader(self):
        fpaths = self.article_dir.glob("*.txt")
        for fpath in fpaths:
            with fpath.open("r") as f:
                article_title = f.name.split("/")[-1][:-4]
                article_body = f.read()

            if self.chunking == "fixed":
                chunks = chunk_string(article_body)
            elif self.chunking == "wiki_sections":
                chunks = wiki_parser(article_body)
            elif self.chunking == "wiki_sections_chunked":
                chunks = wiki_parser(article_body, chunk_sections=True)
            elif self.chunking == "wiki_heading_only":
                chunks = wiki_parser(article_body, heading_only=True)
            else:
                logging.warn(
                    "Chunking type not recognised. Defaulting to fixed length chunking."
                )
                chunks = chunk_string(article_body)

            for i, chunk in enumerate(chunks):
                data_obj = {
                    "title": article_title,
                    "chunk": chunk,
                    "chunk_number": i + 1,
                }

                yield data_obj, None


class JeopardyQuestions1k:
    data_fpath = os.path.join(basedir, "data", "jeopardy_1k.json")
    arr_fpath = os.path.join(basedir, "data", "jeopardy_1k.json.npy")
    category_vec_fpath = os.path.join(basedir, "data", "jeopardy_1k_categories.csv")

    question_collection = "JeopardyQuestion"
    category_collection = "JeopardyCategory"
    xref_prop_name = "hasCategory"

    def add_collections(self, client: WeaviateClient) -> Tuple[Collection, Collection]:
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
                    description="The category title",
                )
            ],
        )

        questions = client.collections.create(
            name=self.question_collection,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            generative_config=Configure.Generative.openai(),
            inverted_index_config=Configure.inverted_index(
                index_property_length=True, index_timestamps=True, index_null_state=True
            ),
            properties=[
                ReferenceProperty(
                    name=self.xref_prop_name,
                    target_collection="JeopardyCategory",
                ),
                Property(
                    name="question",
                    data_type=DataType.TEXT,
                    description="Question asked to the contestant",
                ),
                Property(
                    name="answer",
                    data_type=DataType.TEXT,
                    description="Answer provided by the contestant",
                ),
                Property(
                    name="points", data_type=DataType.INT, description="Jeopardy points"
                ),
                Property(
                    name="round",
                    data_type=DataType.TEXT,
                    description="Jeopardy round",
                    tokenization=Tokenization.FIELD,
                ),
                Property(
                    name="air_date",
                    data_type=DataType.DATE,
                    description="Date that the episode first aired on TV",
                ),
            ],
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

    def upload_objects(self, client: WeaviateClient, batch_size=200) -> bool:
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
                    from_property_name=self.xref_prop_name,
                )

        return True

    def upload_dataset(
        self, client: WeaviateClient, batch_size=300, overwrite=False
    ) -> bool:
        """
        Adds the class to the schema,
        then calls `upload_objects` to upload the objects.
        """
        if type(client) == Client:
            raise TypeError(
                "Sorry, this is for the `v4` Weaviate Python Client, with the WeaviateClient object type. Please refer to the README for more information."
            )

        if overwrite:
            client.collections.delete(self.question_collection)
            client.collections.delete(self.category_collection)

        _ = self.add_collections(client)
        _ = self.upload_objects(client, batch_size=batch_size)
        return True

    def get_sample(self) -> Tuple[Dict, Dict]:
        dl = self._class_pair_dataloader()
        (question_obj, _), (category_obj, _) = next(dl)
        return question_obj, category_obj


class JeopardyQuestions10k(JeopardyQuestions1k):
    data_fpath = os.path.join(basedir, "data", "jeopardy_10k.json")
    arr_fpath = os.path.join(basedir, "data", "jeopardy_10k.json.npy")
    category_vec_fpath = os.path.join(basedir, "data", "jeopardy_10k_categories.csv")
