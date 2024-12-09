import os
import requests
from typing import Dict, Tuple, List, Generator, Literal
from pathlib import Path
import pandas as pd
from weaviate.util import generate_uuid5
import uuid
from weaviate import WeaviateClient, Client
from weaviate.classes.config import (
    Configure,
    Property,
    ReferenceProperty,
    DataType,
    Tokenization,
)
from weaviate.classes.tenants import Tenant
from weaviate.collections.collection import Collection
from weaviate.classes.data import DataReference
from tqdm import tqdm
import numpy as np
import json
import logging
from zipfile import ZipFile


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
    def __init__(
        self,
        collection_name=None,
        vectorizer_config=None,
        generative_config=None,
        mt_config=None,
        tenants=None,
        properties=None,
        inverted_index_config=None,
    ):
        self.collection_name = collection_name or None
        self.vectorizer_config = (
            vectorizer_config or Configure.Vectorizer.text2vec_openai()
        )
        self.generative_config = generative_config or Configure.Generative.openai()
        self.mt_config = mt_config or None
        self.tenants = tenants or []
        self.properties = properties or list()
        self.inverted_index_config = inverted_index_config or Configure.inverted_index(
            index_timestamps=True,
            index_null_state=True,
            index_property_length=True,
        )

        self._basedir = basedir

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
            inverted_index_config=self.inverted_index_config,
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

        def batch_insert(tgt_collection: Collection):
            with tgt_collection.batch.fixed_size(batch_size=batch_size) as batch:
                for data_obj, vector in tqdm(self._class_dataloader()):
                    batch.add_object(
                        properties=data_obj,
                        uuid=generate_uuid5(data_obj),
                        vector=vector,
                    )

        collection = client.collections.get(self.collection_name)
        if self.mt_config is None:
            responses = batch_insert(collection)
        else:
            for tenant in self.tenants:
                tenant_collection = collection.with_tenant(tenant.name)
                responses = batch_insert(tenant_collection)
        return responses

    def upload_dataset(
        self, client: WeaviateClient, batch_size=200, overwrite=False, compress=False
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

        if compress:
            self.vectorindex_config = Configure.VectorIndex.hnsw(
                quantizer=Configure.VectorIndex.Quantizer.bq()
            )
        else:
            self.vectorindex_config = Configure.VectorIndex.hnsw()

        _ = self.add_collection(client)
        upload_responses = self.upload_objects(client, batch_size=batch_size)

        return upload_responses

    def get_sample(self) -> Dict:
        dl = self._class_dataloader()
        data_obj, _ = next(dl)

        return data_obj


class WineReviews(SimpleDataset):
    def __init__(
        self,
        collection_name="WineReview",
        vectorizer_config=None,
        generative_config=None,
    ):
        super().__init__(
            collection_name=collection_name,
            vectorizer_config=vectorizer_config
            or Configure.Vectorizer.text2vec_openai(),
            generative_config=generative_config or Configure.Generative.openai(),
            properties=[
                Property(
                    name="review_body",
                    data_type=DataType.TEXT,
                    description="Review body",
                ),
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    description="Name of the wine",
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
            ],
        )

        # Set Class-specific attributes
        self.winedata_path = os.path.join(self._basedir, "data", "winemag_tiny.csv")

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


class WineReviewsNV(WineReviews):
    def __init__(self):
        super().__init__()
        self.collection_name = "WineReviewNV"
        self.vectorizer_config = [
            Configure.NamedVectors.text2vec_openai(
                name="title", source_properties=["title"]
            ),
            Configure.NamedVectors.text2vec_openai(
                name="review_body", source_properties=["review_body"]
            ),
            Configure.NamedVectors.text2vec_openai(
                name="title_country", source_properties=["title", "country"]
            ),
        ]


class Wiki100(SimpleDataset):
    def __init__(
        self,
        collection_name="WikiChunk",
        vectorizer_config=None,
        generative_config=None,
    ):
        super().__init__(
            collection_name=collection_name,
            vectorizer_config=vectorizer_config
            or Configure.Vectorizer.text2vec_openai(),
            generative_config=generative_config or Configure.Generative.openai(),
            properties=[
                Property(
                    name="title", data_type=DataType.TEXT, description="Article title"
                ),
                Property(
                    name="chunk", data_type=DataType.TEXT, description="Text chunk"
                ),
                Property(
                    name="chunk_number",
                    data_type=DataType.INT,
                    description="Chunk number - 1 index",
                ),
            ],
        )

        # Set Class-specific attributes
        self._basedir = basedir
        self.article_dir = Path(self._basedir) / "data/wiki100"
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
                article_title = fpath.stem
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
    def __init__(
        self, vectorizer_config=None, generative_config=None, reranker_config=None
    ):
        self.vectorizer_config = (
            vectorizer_config or Configure.Vectorizer.text2vec_openai()
        )
        self.generative_config = generative_config or Configure.Generative.openai()
        self.reranker_config = reranker_config or Configure.Reranker.cohere()

        self._basedir = basedir

        self._data_fpath = os.path.join(self._basedir, "data", "jeopardy_1k.json")
        self._arr_fpath = os.path.join(self._basedir, "data", "jeopardy_1k.json.npy")
        self._category_vec_fpath = os.path.join(
            self._basedir, "data", "jeopardy_1k_categories.csv"
        )

        self._question_collection = "JeopardyQuestion"
        self._category_collection = "JeopardyCategory"
        self._xref_prop_name = "hasCategory"

        if vectorizer_config is not None:
            self._use_existing_vecs = False
        else:
            self._use_existing_vecs = True


    def add_collections(self, client: WeaviateClient) -> Tuple[Collection, Collection]:
        """
        For each class in the dataset, add its definition to the Weaviate instance.
        """
        categories = client.collections.create(
            name=self._category_collection,
            vectorizer_config=self.vectorizer_config,
            vector_index_config=self.vectorindex_config,
            generative_config=self.generative_config,
            reranker_config=self.reranker_config,
            properties=[
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    description="The category title",
                )
            ],
        )

        questions = client.collections.create(
            name=self._question_collection,
            vectorizer_config=self.vectorizer_config,
            vector_index_config=self.vectorindex_config,
            generative_config=self.generative_config,
            reranker_config=self.reranker_config,
            inverted_index_config=Configure.inverted_index(
                index_property_length=True, index_timestamps=True, index_null_state=True
            ),
            properties=[
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
            references=[
                ReferenceProperty(
                    name=self._xref_prop_name,
                    target_collection="JeopardyCategory",
                ),
            ],
        )
        return categories, questions

    def _class_pair_dataloader(self):
        from datetime import datetime, timezone

        question_vec_array = np.load(self._arr_fpath)
        category_vec_dict = self._get_cat_array()

        with open(self._data_fpath, "r") as f:
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
        cat_df = pd.read_csv(self._category_vec_fpath)
        cat_arr = cat_df.iloc[:, :-1].to_numpy()
        cat_names = cat_df["category"].to_list()
        cat_emb_dict = dict(zip(cat_names, cat_arr))
        return cat_emb_dict

    def upload_objects(self, client: WeaviateClient) -> bool:
        """
        Base uploader method for uploading a single class.
        """
        with client.batch.fixed_size() as batch:
            for (data_obj_from, vec_from), (data_obj_to, vec_to) in tqdm(
                self._class_pair_dataloader()
            ):
                # Use existing vectors if available
                if not self._use_existing_vecs:
                    vec_from = None
                    vec_to = None

                # Add "class_from" objects
                id_from = generate_uuid5(data_obj_from)

                batch.add_object(
                    properties=data_obj_from,
                    collection=self._question_collection,
                    uuid=id_from,
                    vector=vec_from,
                )

                # Add "class_to" objects
                id_to = generate_uuid5(data_obj_to)
                batch.add_object(
                    properties=data_obj_to,
                    collection=self._category_collection,
                    uuid=id_to,
                    vector=vec_to,
                )

                # Add references
                batch.add_reference(
                    from_collection=self._question_collection,
                    from_uuid=id_from,
                    from_property=self._xref_prop_name,
                    to=id_to,
                )

        return True

    def upload_dataset(
        self, client: WeaviateClient, overwrite=False, compress=False
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
            client.collections.delete(self._question_collection)
            client.collections.delete(self._category_collection)

        if compress:
            self.vectorindex_config = Configure.VectorIndex.hnsw(
                quantizer=Configure.VectorIndex.Quantizer.bq()
            )
        else:
            self.vectorindex_config = Configure.VectorIndex.hnsw()

        _ = self.add_collections(client)
        _ = self.upload_objects(client)
        return True

    def get_sample(self) -> Tuple[Dict, Dict]:
        dl = self._class_pair_dataloader()
        (question_obj, _), (category_obj, _) = next(dl)
        return question_obj, category_obj


class JeopardyQuestions10k(JeopardyQuestions1k):
    def __init__(
        self, vectorizer_config=None, generative_config=None, reranker_config=None
    ):
        super().__init__( vectorizer_config, generative_config, reranker_config)
        self.data_fpath = os.path.join(self._basedir, "data", "jeopardy_10k.json")
        self.arr_fpath = os.path.join(self._basedir, "data", "jeopardy_10k.json.npy")
        self.category_vec_fpath = os.path.join(
            self._basedir, "data", "jeopardy_10k_categories.csv"
        )

# class NewsArticles(SimpleDataset):

#     # Not sure if worth the effort required to port this over from the V3 client

#     _embeddings_files = {
#         "articles": "data/newsarticles_Article_openai_embeddings.json",
#         "authors": "data/newsarticles_Author_openai_embeddings.json",
#         "categories": "data/newsarticles_Category_openai_embeddings.json",
#         "publications": "data/newsarticles_Publication_openai_embeddings.json",
#     }

#     _datadir = os.path.join(basedir, "data/newsarticles")

#     def __init__(
#         self,
#         # generative_config=None,
#     ):
#         super().__init__()
#         self._dataset_size = None
#         self._dataset_path = os.path.join(basedir, "data/newsarticles.zip")
#         self.vectorizer_config = Configure.Vectorizer.text2vec_openai()
#         with open(os.path.join(basedir, "data/newsarticles_schema.json")) as f:
#             self._class_definitions = json.load(f)

#         # Download the dataset if not done so already
#         if not os.path.exists(self._dataset_path):
#             ## Download data https://github.com/databyjp/wv_demo_uploader/raw/main/weaviate_datasets/data/newsarticles.zip
#             print("Downloading data... please wait")
#             url = "https://github.com/databyjp/wv_demo_uploader/raw/main/weaviate_datasets/data/newsarticles.zip"
#             r = requests.get(url)
#             with open(self._dataset_path, "wb") as f:
#                 f.write(r.content)

#         # unzip the data if not done so already
#         if not os.path.exists(Path(basedir) / "data" / "newsarticles"):
#             print("Unzipping data...")
#             zipfile = self._dataset_path
#             with ZipFile(zipfile, "r") as zip_ref:
#                 zip_ref.extractall(os.path.join(basedir, "data"))

#     def add_to_schema(self, client: WeaviateClient) -> str:
#         for c in self._class_definitions["classes"]:
#             response = client.collections.create_from_dict(c)
#         return str(response)

#     def upload_dataset(self, client: WeaviateClient, batch_size=300, overwrite=False) -> bool:
#         if overwrite:
#             for coll_definition in self._class_definitions["classes"]:
#                 client.collections.delete(coll_definition["class"])

#         self.add_to_schema(client)
#         self._load_publication_and_category(client, batch_size)
#         self._load_authors_article(client, batch_size)
#         return True


#     def _get_sub_filelist(self, filedir):
#         return [f for f in os.listdir(filedir) if f.endswith(".json")]

#     def _load_publication_and_category(self, client: WeaviateClient, batch_size: int = 100):
#         for ctype in ["categories", "publications"]:
#             datafiles = self._get_sub_filelist(os.path.join(self._datadir, ctype))
#             embeddings_file = NewsArticles._embeddings_files[ctype]
#             with open(os.path.join(basedir, embeddings_file), "r") as f:
#                 embeddings = json.load(f)

#             with client.batch.fixed_size(batch_size=batch_size) as batch:
#                 for dfile in datafiles:
#                     with open(os.path.join(self._datadir, ctype, dfile), "r") as f:
#                         data = json.load(f)
#                     batch.add_object(
#                         properties=data["schema"],
#                         collection=data["class"],
#                         uuid=data["id"],
#                         vector=embeddings[data["id"]],
#                     )

#     def _load_authors_article(self, client: WeaviateClient, batch_size: int = 50):
#         datafiles = self._get_sub_filelist(os.path.join(self._datadir))
#         embedding_dict = {}
#         for ctype in ["articles", "authors"]:
#             embeddings_file = NewsArticles._embeddings_files[ctype]
#             with open(os.path.join(basedir, embeddings_file), "r") as f:
#                 embeddings = json.load(f)
#             embedding_dict[ctype] = embeddings

#         with client.batch.fixed_size(batch_size=batch_size) as batch:
#             for datafile in datafiles:
#                 try:
#                     with open(os.path.join(self._datadir, datafile), "r") as f:
#                         data = json.load(f)

#                     article_id = NewsArticles._generate_uuid(data["url"])

#                     #### ADD AUTHORS #####
#                     author_ids = []
#                     for author in data["authors"]:
#                         if len(author.split(" ")) == 2:
#                             author = NewsArticles._clean_up_newsdata("Author", author)
#                             author_id = NewsArticles._generate_uuid(author)
#                             if author_id in embeddings.keys():
#                                 batch.add_object(
#                                     properties={"name": author},
#                                     collection="Author",
#                                     uuid=author_id,
#                                     vector=embeddings[author_id],
#                                 )
#                                 author_ids.append(author_id)
#                                 batch.add_reference(
#                                     from_uuid=author_id,
#                                     from_collection="Author",
#                                     from_property="writesFor",
#                                     to=data["publicationId"],
#                                 )
#                                 batch.add_reference(
#                                     from_uuid=author_id,
#                                     from_collection="Author",
#                                     from_property="wroteArticles",
#                                     to=article_id,
#                                 )
#                         else:
#                             author_id = data["publicationId"]
#                             author_ids.append(data["publicationId"])

#                     ##### ADD ARTICLES #####

#                     word_count = len(" ".join(data["paragraphs"]).split(" "))
#                     article_object = {
#                         "title": data["title"],
#                         "summary": NewsArticles._clean_up_newsdata(
#                             "Summary", data["summary"]
#                         ),
#                         "wordCount": word_count,
#                         "url": data["url"],
#                     }
#                     # Set publication date
#                     if data["pubDate"] is not None and data["pubDate"] != "":
#                         article_object["publicationDate"] = data["pubDate"]
#                     # Add article to weaviate
#                     batch.add_object(
#                         properties=article_object,
#                         collection="Article",
#                         uuid=article_id,
#                         vector=embedding_dict["articles"][article_id],
#                     )

#                     article_id = NewsArticles._generate_uuid(data["url"])

#                     # Add reference to weaviate
#                     batch.add_reference(
#                         from_uuid=article_id,
#                         from_collection="Article",
#                         from_property="inPublication",
#                         to=data["publicationId"],
#                     )
#                     batch.add_reference(
#                         from_uuid=data["publicationId"],
#                         from_collection="Publication",
#                         from_property="hasArticles",
#                         to=article_id,
#                     )

#                     for author_id in author_ids:
#                         batch.add_reference(
#                             from_uuid=article_id,
#                             from_collection="Article",
#                             from_property="hasAuthors",
#                             to=author_id,
#                         )
#                 except Exception as e:
#                     print(f"Error while loading {datafile}: {e}")

#     def _generate_uuid(key: str) -> str:
#         """
#         Generates an universally unique identifier (uuid).
#         Note: This is an older version of the uuid generation used in this dataset only.
#         For new data, use weaviate.util.generate_uuid5() instead.

#         Parameters
#         ----------
#         key : str
#             The key used to generate the uuid.

#         Returns
#         -------
#         str
#             Universally unique identifier (uuid) as string.
#         """

#         return str(uuid.uuid3(uuid.NAMESPACE_DNS, key))

#     def _clean_up_newsdata(class_name: str, value: str) -> str:
#         """
#         Clean up the data.

#         Parameters
#         ----------
#         class_name: str
#             Which class the object(see value) to clean belongs to.
#         value: str
#             The object to clean.

#         Returns
#         -------
#         str
#             Cleaned object.
#         """

#         if class_name == "Author":
#             value = value.replace(" Wsj.Com", "")
#             value = value.replace(".", " ")
#         elif class_name == "Summary":
#             value = value.replace("\n", " ")
#         return value
