import os
import json
from typing import Dict

import pandas as pd
import numpy as np
from weaviate.util import generate_uuid5
import uuid
from weaviate import Client, WeaviateClient
from weaviate.schema import Tenant
from tqdm import tqdm
import logging
from pathlib import Path
from zipfile import ZipFile
import requests


logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

basedir = os.path.dirname(os.path.abspath(__file__))


class Dataset:

    """
    Base class for all datasets.
    """

    def __init__(self):
        """
        Initializes the class definition and dataset size.
        """
        self._class_definitions = []
        self._dataset_size = None

    def get_class_definitions(self) -> list:
        """
        Load class definitions to be added to the schema.
        """
        return self._class_definitions

    def get_class_names(self) -> list:
        """
        Get class names in the schema.
        """
        return [c["class"] for c in self._class_definitions]

    def get_dataset_size(self) -> int:
        """
        Return the size of the dataset.
        """
        try:
            return self._dataset_size
        except:
            return None

    def get_sample(self) -> None:
        """
        Placeholder method for returning a sample of the dataset.
        To be implemented in child classes.
        """
        return {}

    # ----- METHODS FOR SCHEMA DETAILS -----

    def _class_in_schema(self, client: Client, class_name) -> bool:
        """
        For a given class name, checks if it is present in the Weaviate schema.
        """
        schema = client.schema.get()
        return class_name in [wv_class["class"] for wv_class in schema["classes"]]

    def classes_in_schema(self, client: Client) -> dict:
        """
        Polls the Weaviate instance to check if any of the classes to be populated already exists.
        """
        class_names = self.get_class_names()
        return {
            class_name: self._class_in_schema(client, class_name)
            for class_name in class_names
        }

    def delete_existing_dataset_classes(self, client: Client) -> bool:
        """
        Delete classes to be populated the Weaviate instance.
        """
        class_names = self.get_class_names()
        for class_name in class_names:
            if self._class_in_schema(client, class_name):
                client.schema.delete_class(class_name)
        return True

    def add_to_schema(self, client: Client) -> bool:
        """
        For each class in the dataset, add its definition to the Weaviate instance.
        """
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
        """
        Update the class definition with a new vectorizer name and module config.
        """
        # TODO - add vectorizer / module config validation here
        # TODO - check if property-level moduleConfig settings need to be renamed accordingly also

        for i, class_def in enumerate(self._class_definitions):
            class_def["vectorizer"] = vectorizer_name
            class_def["moduleConfig"] = module_config
        return self.get_class_definitions()

    # ----- DATA UPLOADER GENERIC METHODS -----

    def _class_uploader(
        self, client: Client, class_name: str, batch_size: int, tenant=None
    ) -> bool:
        """
        Base uploader method for uploading a single class.
        """
        client.batch.configure(batch_size=batch_size)
        with client.batch as batch:
            for data_obj, vector in tqdm(self._class_dataloader(class_name)):
                uuid = generate_uuid5(data_obj)
                batch.add_data_object(
                    data_obj,
                    class_name,
                    uuid=uuid,
                    vector=vector,
                    tenant=tenant,
                )

        return True

    def _class_pair_uploader(
        self,
        client: Client,
        class_from: str,
        class_to: str,
        batch_size: int,
        tenant=None,
    ) -> bool:
        """
        Base uploader method for uploading a pair of classes.
        """
        if tenant is not None:
            for class_name in [class_from, class_to]:
                client.schema.add_class_tenants(
                    class_name=class_name, tenants=[Tenant(name=tenant)]
                )

        client.batch.configure(batch_size=batch_size)
        with client.batch as batch:
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
                    tenant=tenant,
                )
                # Add "class_to" objects
                id_to = generate_uuid5(data_obj_to)
                batch.add_data_object(
                    data_obj_to, class_to, uuid=id_to, vector=vec_to, tenant=tenant
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
                        tenant=tenant,
                    )

        return True

    def upload_dataset(self, client: Client, batch_size=300) -> bool:
        """
        Adds the class to the schema,
        then calls `upload_objects` to upload the objects.
        """
        if type(client) == WeaviateClient:
            raise TypeError(
                "Sorry, this is for the `v3` Weaviate Python Client, with the Client object type. Please refer to the README for more information."
            )

        schema_add_results = self.add_to_schema(client)
        self.upload_objects(client, batch_size=batch_size)
        return True


class WineReviews(Dataset):
    winedata_path = os.path.join(basedir, "data", "winemag_tiny.csv")
    class_name = "WineReview"

    def __init__(self):
        super().__init__()
        self._dataset_size = len(pd.read_csv(self.winedata_path))
        self._class_definitions = [
            {
                "class": self.class_name,
                "vectorizer": "text2vec-openai",
                "moduleConfig": {
                    "generative-openai": {
                        "model": "gpt-3.5-turbo",
                    }
                },
                "properties": [
                    {
                        "name": "review_body",
                        "dataType": ["text"],
                        "description": "Review body",
                    },
                    {
                        "name": "title",
                        "dataType": ["text"],
                        "description": "Name of the wine",
                    },
                    {
                        "name": "country",
                        "dataType": ["text"],
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
        if class_name == self.class_name:
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
        else:
            raise ValueError("Unknown class name")

    def upload_objects(self, client: Client, batch_size: int) -> bool:
        for class_name in self.get_class_names():
            self._class_uploader(client, class_name, batch_size)
        return True

    def get_sample(self) -> dict:
        samples = dict()
        for c in self.get_class_names():
            dl = self._class_dataloader(c)
            samples[c] = next(dl)
        return samples


class WineReviewsMT(WineReviews):
    winedata_path = os.path.join(basedir, "data", "winemag_tiny.csv")
    tenants = ["tenantA", "tenantB"]
    class_name = "WineReviewMT"

    def __init__(self):
        super().__init__()
        for i in range(len(self._class_definitions)):
            self._class_definitions[i]["multiTenancyConfig"] = {"enabled": True}

    def upload_objects(self, client: Client, batch_size: int) -> bool:
        for tenant in self.tenants:
            for class_name in self.get_class_names():
                client.schema.add_class_tenants(
                    class_name=class_name, tenants=[Tenant(name=tenant)]
                )
                self._class_uploader(client, class_name, batch_size, tenant)
        return True

    def get_sample(self) -> dict:
        samples = dict()
        for c in self.get_class_names():
            dl = self._class_dataloader(c)
            samples[c] = next(dl)
        return samples


class JeopardyQuestions1k(Dataset):
    data_fpath = os.path.join(basedir, "data", "jeopardy_1k.json")
    arr_fpath = os.path.join(basedir, "data", "jeopardy_1k.json.npy")
    category_vec_fpath = os.path.join(basedir, "data", "jeopardy_1k_categories.csv")

    question_class = "JeopardyQuestion"
    category_class = "JeopardyCategory"

    def __init__(self):
        super().__init__()
        self._dataset_size = 1000
        self._class_definitions = [
            {
                "class": self.category_class,
                "description": "A Jeopardy! category",
                "vectorizer": "text2vec-openai",
                "properties": [
                    {
                        "name": "title",
                        "dataType": ["text"],
                        "description": "The category title",
                    },
                ],
                "moduleConfig": {
                    "generative-openai": {
                        "model": "gpt-3.5-turbo",
                    }
                },
            },
            {
                "class": self.question_class,
                "description": "A Jeopardy! question",
                "vectorizer": "text2vec-openai",
                "moduleConfig": {
                    "generative-openai": {
                        "model": "gpt-3.5-turbo",
                    }
                },
                "invertedIndexConfig": {
                    "indexPropertyLength": True,
                    "indexTimestamps": True,
                    "indexNullState": True,
                },
                "properties": [
                    {
                        "name": "hasCategory",
                        "dataType": [self.category_class],
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
                        "name": "points",
                        "dataType": ["int"],
                        "description": "Points that the question was worth",
                    },
                    {
                        "name": "round",
                        "dataType": ["text"],
                        "description": "Jeopardy round",
                        "tokenization": "field",
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

    def upload_objects(self, client: Client, batch_size: int) -> bool:
        return self._class_pair_uploader(
            client,
            class_from=self.question_class,
            class_to=self.category_class,
            batch_size=batch_size,
        )

    def get_sample(self) -> dict:
        samples = dict()
        dl = self._class_pair_dataloader()
        (question_obj, question_vec), (category_obj, category_vec) = next(dl)
        samples[self.category_class] = category_obj
        samples[self.question_class] = question_obj
        return samples


class JeopardyQuestions10k(JeopardyQuestions1k):
    data_fpath = os.path.join(basedir, "data", "jeopardy_10k.json")
    arr_fpath = os.path.join(basedir, "data", "jeopardy_10k.json.npy")
    category_vec_fpath = os.path.join(basedir, "data", "jeopardy_10k_categories.csv")

    def __init__(self):
        super().__init__()
        self._dataset_size = 10000


class JeopardyQuestions1kMT(JeopardyQuestions1k):
    tenants = ["tenantA", "tenantB"]

    question_class = "JeopardyQuestionMT"
    category_class = "JeopardyCategoryMT"

    def __init__(self):
        super().__init__()
        for i in range(len(self._class_definitions)):
            self._class_definitions[i]["multiTenancyConfig"] = {"enabled": True}

    def upload_objects(self, client: Client, batch_size: int) -> bool:
        for tenant in self.tenants:
            self._class_pair_uploader(
                client,
                class_from=self.question_class,
                class_to=self.category_class,
                batch_size=batch_size,
                tenant=tenant,
            )
        return True


class NewsArticles(Dataset):
    _embeddings_files = {
        "articles": "data/newsarticles_Article_openai_embeddings.json",
        "authors": "data/newsarticles_Author_openai_embeddings.json",
        "categories": "data/newsarticles_Category_openai_embeddings.json",
        "publications": "data/newsarticles_Publication_openai_embeddings.json",
    }

    _datadir = os.path.join(basedir, "data/newsarticles")

    def __init__(self):
        super().__init__()
        self._dataset_size = None
        self._dataset_path = os.path.join(basedir, "data/newsarticles.zip")
        with open(os.path.join(basedir, "data/newsarticles_schema.json")) as f:
            self._class_definitions = json.load(f)

        # Download the dataset if not done so already
        if not os.path.exists(self._dataset_path):
            ## Download data https://github.com/databyjp/wv_demo_uploader/raw/main/weaviate_datasets/data/newsarticles.zip
            print("Downloading data... please wait")
            url = "https://github.com/databyjp/wv_demo_uploader/raw/main/weaviate_datasets/data/newsarticles.zip"
            r = requests.get(url)
            with open(self._dataset_path, "wb") as f:
                f.write(r.content)

        # unzip the data if not done so already
        if not os.path.exists(Path(basedir) / "data" / "newsarticles"):
            print("Unzipping data...")
            zipfile = self._dataset_path
            with ZipFile(zipfile, "r") as zip_ref:
                zip_ref.extractall(os.path.join(basedir, "data"))

    def add_to_schema(self, client: Client) -> str:
        response = client.schema.create(self._class_definitions)
        return str(response)

    def upload_dataset(self, client: Client, batch_size=300) -> bool:
        if type(client) == WeaviateClient:
            raise TypeError(
                "Sorry, this is for the `v3` Weaviate Python Client, with the Client object type. Please refer to the README for more information."
            )

        self.add_to_schema(client)
        self._load_publication_and_category(client, batch_size)
        self._load_authors_article(client, batch_size)
        return True

    def _get_sub_filelist(self, filedir):
        return [f for f in os.listdir(filedir) if f.endswith(".json")]

    def _load_publication_and_category(self, client: Client, batch_size: int = 100):
        for ctype in ["categories", "publications"]:
            datafiles = self._get_sub_filelist(os.path.join(self._datadir, ctype))
            embeddings_file = NewsArticles._embeddings_files[ctype]
            with open(os.path.join(basedir, embeddings_file), "r") as f:
                embeddings = json.load(f)

            client.batch.configure(batch_size=batch_size)
            with client.batch as batch:
                for dfile in datafiles:
                    with open(os.path.join(self._datadir, ctype, dfile), "r") as f:
                        data = json.load(f)
                    batch.add_data_object(
                        data_object=data["schema"],
                        class_name=data["class"],
                        uuid=data["id"],
                        vector=embeddings[data["id"]],
                    )

    def _load_authors_article(self, client: Client, batch_size: int = 50):
        datafiles = self._get_sub_filelist(os.path.join(self._datadir))
        embedding_dict = {}
        for ctype in ["articles", "authors"]:
            embeddings_file = NewsArticles._embeddings_files[ctype]
            with open(os.path.join(basedir, embeddings_file), "r") as f:
                embeddings = json.load(f)
            embedding_dict[ctype] = embeddings

        client.batch.configure(batch_size=batch_size)
        with client.batch as batch:
            for datafile in datafiles:
                try:
                    with open(os.path.join(self._datadir, datafile), "r") as f:
                        data = json.load(f)

                    article_id = NewsArticles._generate_uuid(data["url"])

                    #### ADD AUTHORS #####
                    author_ids = []
                    for author in data["authors"]:
                        if len(author.split(" ")) == 2:
                            author = NewsArticles._clean_up_newsdata("Author", author)
                            author_id = NewsArticles._generate_uuid(author)
                            if author_id in embeddings.keys():
                                batch.add_data_object(
                                    data_object={"name": author},
                                    class_name="Author",
                                    uuid=author_id,
                                    vector=embeddings[author_id],
                                )
                                author_ids.append(author_id)
                                batch.add_reference(
                                    from_object_uuid=author_id,
                                    from_object_class_name="Author",
                                    from_property_name="writesFor",
                                    to_object_uuid=data["publicationId"],
                                    to_object_class_name="Publication",
                                )
                                batch.add_reference(
                                    from_object_uuid=author_id,
                                    from_object_class_name="Author",
                                    from_property_name="wroteArticles",
                                    to_object_uuid=article_id,
                                    to_object_class_name="Article",
                                )
                        else:
                            author_id = data["publicationId"]
                            author_ids.append(data["publicationId"])

                    ##### ADD ARTICLES #####

                    word_count = len(" ".join(data["paragraphs"]).split(" "))
                    article_object = {
                        "title": data["title"],
                        "summary": NewsArticles._clean_up_newsdata(
                            "Summary", data["summary"]
                        ),
                        "wordCount": word_count,
                        "url": data["url"],
                    }
                    # Set publication date
                    if data["pubDate"] is not None and data["pubDate"] != "":
                        article_object["publicationDate"] = data["pubDate"]
                    # Add article to weaviate
                    batch.add_data_object(
                        data_object=article_object,
                        class_name="Article",
                        uuid=article_id,
                        vector=embedding_dict["articles"][article_id],
                    )

                    article_id = NewsArticles._generate_uuid(data["url"])

                    # Add reference to weaviate
                    batch.add_reference(
                        from_object_uuid=article_id,
                        from_object_class_name="Article",
                        from_property_name="inPublication",
                        to_object_uuid=data["publicationId"],
                        to_object_class_name="Publication",
                    )
                    batch.add_reference(
                        from_object_uuid=data["publicationId"],
                        from_object_class_name="Publication",
                        from_property_name="hasArticles",
                        to_object_uuid=article_id,
                        to_object_class_name="Article",
                    )

                    for author_id in author_ids:
                        batch.add_reference(
                            from_object_uuid=article_id,
                            from_object_class_name="Article",
                            from_property_name="hasAuthors",
                            to_object_uuid=author_id,
                            to_object_class_name="Author",
                        )
                except Exception as e:
                    print(f"Error while loading {datafile}: {e}")

    def _generate_uuid(key: str) -> str:
        """
        Generates an universally unique identifier (uuid).
        Note: This is an older version of the uuid generation used in this dataset only.
        For new data, use weaviate.util.generate_uuid5() instead.

        Parameters
        ----------
        key : str
            The key used to generate the uuid.

        Returns
        -------
        str
            Universally unique identifier (uuid) as string.
        """

        return str(uuid.uuid3(uuid.NAMESPACE_DNS, key))

    def _clean_up_newsdata(class_name: str, value: str) -> str:
        """
        Clean up the data.

        Parameters
        ----------
        class_name: str
            Which class the object(see value) to clean belongs to.
        value: str
            The object to clean.

        Returns
        -------
        str
            Cleaned object.
        """

        if class_name == "Author":
            value = value.replace(" Wsj.Com", "")
            value = value.replace(".", " ")
        elif class_name == "Summary":
            value = value.replace("\n", " ")
        return value
