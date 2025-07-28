import os
import requests
from typing import Dict, Tuple, List, Generator, Literal
from pathlib import Path
import pandas as pd
from weaviate.util import generate_uuid5
import uuid
from weaviate import WeaviateClient
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
        vector_config=None,
        generative_config=None,
        mt_config=None,
        tenants=None,
        properties=None,
        inverted_index_config=None,
    ):
        self.collection_name = collection_name or None
        self._vector_config = vector_config or Configure.Vectors.text2vec_openai()
        self._generative_config = generative_config or Configure.Generative.openai()
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
            vector_config=self._vector_config,
            generative_config=self._generative_config,
            properties=self.properties,
            multi_tenancy_config=self.mt_config,
            inverted_index_config=self.inverted_index_config,
        )
        if self.mt_config is not None:
            collection.tenants.create(self.tenants)

        return collection

    def _class_dataloader(self) -> Generator:
        yield {}, None

    def upload_objects(self, client: WeaviateClient, batch_size: int = 200) -> List:
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

        if overwrite:
            client.collections.delete(self.collection_name)

        # Configure vector index based on compression setting
        from weaviate.collections.classes.config import _VectorConfigCreate
        if not isinstance(self._vector_config, _VectorConfigCreate):
            # vector_config is a list
            if compress:
                self._vector_config[0].vector_index_config = Configure.VectorIndex.hnsw(
                    quantizer=Configure.VectorIndex.Quantizer.bq()
                )
            else:
                self._vector_config[0].vector_index_config = Configure.VectorIndex.hnsw()
        else:
            # vector_config is a single object
            if compress:
                self._vector_config.vectorIndexConfig = Configure.VectorIndex.hnsw(
                    quantizer=Configure.VectorIndex.Quantizer.bq()
                )
            else:
                self._vector_config.vectorIndexConfig = Configure.VectorIndex.hnsw()

        _ = self.add_collection(client)
        upload_responses = self.upload_objects(client, batch_size)

        return upload_responses

    def get_sample(self) -> Dict:
        dl = self._class_dataloader()
        data_obj, _ = next(dl)

        return data_obj


class WineReviews(SimpleDataset):
    def __init__(
        self,
        collection_name="WineReview",
        vector_config=None,
        generative_config=None,
    ):
        super().__init__(
            collection_name=collection_name,
            vector_config=vector_config or Configure.Vectors.text2vec_openai(),
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
        self.vector_config = [
            Configure.Vectors.text2vec_openai(
                name="title", source_properties=["title"]
            ),
            Configure.Vectors.text2vec_openai(
                name="review_body", source_properties=["review_body"]
            ),
            Configure.Vectors.text2vec_openai(
                name="title_country", source_properties=["title", "country"]
            ),
        ]


class Wiki100(SimpleDataset):
    def __init__(
        self,
        collection_name="WikiChunk",
        vector_config=None,
        generative_config=None,
    ):
        super().__init__(
            collection_name=collection_name,
            vector_config=vector_config or Configure.Vectors.text2vec_openai(),
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
        self, vector_config=None, generative_config=None, reranker_config=None
    ):

        if vector_config is None:
            vector_config = Configure.Vectors.text2vec_openai(
                model="ada", model_version="002", type_="text"
            )
            self._use_existing_vecs = True
        else:
            self._use_existing_vecs = False

        if generative_config is None:
            generative_config = Configure.Generative.openai()

        if reranker_config is None:
            reranker_config = Configure.Reranker.cohere()

        self._basedir = basedir

        self._data_fpath = os.path.join(self._basedir, "data", "jeopardy_1k.json")
        self._arr_fpath = os.path.join(self._basedir, "data", "jeopardy_1k.json.npy")
        self._category_vec_fpath = os.path.join(
            self._basedir, "data", "jeopardy_1k_categories.csv"
        )

        self._question_collection = "JeopardyQuestion"
        self._category_collection = "JeopardyCategory"
        self._xref_prop_name = "hasCategory"

        self._vector_config = vector_config
        self._generative_config = generative_config
        self._reranker_config = reranker_config

    def add_collections(self, client: WeaviateClient) -> Tuple[Collection, Collection]:
        """
        For each class in the dataset, add its definition to the Weaviate instance.
        """
        categories = client.collections.create(
            name=self._category_collection,
            vector_config=self._vector_config,
            generative_config=self._generative_config,
            reranker_config=self._reranker_config,
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
            vector_config=self._vector_config,
            generative_config=self._generative_config,
            reranker_config=self._reranker_config,
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

    def upload_objects(self, client: WeaviateClient, batch_size: int = 200) -> bool:
        """
        Base uploader method for uploading a single class.
        """
        with client.batch.fixed_size(batch_size=batch_size) as batch:
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
        self, client: WeaviateClient, overwrite=False, compress=False, batch_size=200
    ) -> bool:
        """
        Adds the class to the schema,
        then calls `upload_objects` to upload the objects.
        """

        if overwrite:
            client.collections.delete(self._question_collection)
            client.collections.delete(self._category_collection)

        # Configure vector index based on compression setting
        from weaviate.collections.classes.config import _VectorConfigCreate
        if not isinstance(self._vector_config, _VectorConfigCreate):
            # vector_config is a list
            if compress:
                self._vector_config[0].vector_index_config = Configure.VectorIndex.hnsw(
                    quantizer=Configure.VectorIndex.Quantizer.bq()
                )
            else:
                self._vector_config[0].vector_index_config = Configure.VectorIndex.hnsw()
        else:
            # vector_config is a single object
            if compress:
                self._vector_config.vectorIndexConfig = Configure.VectorIndex.hnsw(
                    quantizer=Configure.VectorIndex.Quantizer.bq()
                )
            else:
                self._vector_config.vectorIndexConfig = Configure.VectorIndex.hnsw()

        _ = self.add_collections(client)
        _ = self.upload_objects(client, batch_size)
        return True

    def get_sample(self) -> Tuple[Dict, Dict]:
        dl = self._class_pair_dataloader()
        (question_obj, _), (category_obj, _) = next(dl)
        return question_obj, category_obj


class JeopardyQuestions10k(JeopardyQuestions1k):
    def __init__(
        self, vector_config=None, generative_config=None, reranker_config=None
    ):
        super().__init__(vector_config, generative_config, reranker_config)
        self._data_fpath = os.path.join(self._basedir, "data", "jeopardy_10k.json")
        self._arr_fpath = os.path.join(self._basedir, "data", "jeopardy_10k.json.npy")
        self._category_vec_fpath = os.path.join(
            self._basedir, "data", "jeopardy_10k_categories.csv"
        )


class NewsArticles(SimpleDataset):
    _embeddings_files = {
        "articles": "data/newsarticles_Article_openai_embeddings.json",
        "authors": "data/newsarticles_Author_openai_embeddings.json",
        "categories": "data/newsarticles_Category_openai_embeddings.json",
        "publications": "data/newsarticles_Publication_openai_embeddings.json",
    }

    def __init__(
        self,
        collection_name=None,  # Not used but kept for consistency with parent class
        vector_config=None,
        generative_config=None,
    ):
        super().__init__(
            collection_name=collection_name,
            vector_config=vector_config or Configure.Vectors.text2vec_openai(),
            generative_config=generative_config or Configure.Generative.openai(),
        )

        self._datadir = os.path.join(self._basedir, "data/newsarticles")
        self._dataset_size = None
        self._dataset_path = os.path.join(self._basedir, "data/newsarticles.zip")

        # Download the dataset if not done so already
        if not os.path.exists(self._dataset_path):
            print("Downloading data... please wait")
            url = "https://github.com/databyjp/wv_demo_uploader/raw/main/weaviate_datasets/data/newsarticles.zip"
            r = requests.get(url)
            with open(self._dataset_path, "wb") as f:
                f.write(r.content)

        # unzip the data if not done so already
        if not os.path.exists(Path(self._basedir) / "data" / "newsarticles"):
            print("Unzipping data...")
            zipfile = self._dataset_path
            with ZipFile(zipfile, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self._basedir, "data"))

    def add_collections(self, client: WeaviateClient) -> dict:
        """
        Add all collections (Publication, Author, Article, Category) to the schema
        """
        collections = {}

        # First, create the collections with no cross-references
        # Category collection (no references needed)
        collections["Category"] = client.collections.create(
            name="Category",
            description="Category an article belongs to",
            vector_config=self.vector_config,
            generative_config=self._generative_config,  # Fixed: use _generative_config
            properties=[
                Property(
                    name="name",
                    data_type=DataType.TEXT,
                    description="Category name",
                    tokenization=Tokenization.FIELD,
                ),
            ],
        )

        # Article collection (no references initially)
        collections["Article"] = client.collections.create(
            name="Article",
            description="A news article",
            vector_config=self.vector_config,
            generative_config=self._generative_config,
            inverted_index_config=Configure.inverted_index(
                index_timestamps=True,
                index_null_state=True,
                index_property_length=True,
            ),
            properties=[
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    description="Title of the article",
                    tokenization=Tokenization.WORD,
                ),
                Property(
                    name="url",
                    data_type=DataType.TEXT,
                    description="The url of the article",
                    tokenization=Tokenization.FIELD,
                ),
                Property(
                    name="summary",
                    data_type=DataType.TEXT,
                    description="The summary of the article",
                    tokenization=Tokenization.WORD,
                ),
                Property(
                    name="publicationDate",
                    data_type=DataType.DATE,
                    description="The date of publication of the article",
                ),
                Property(
                    name="wordCount",
                    data_type=DataType.INT,
                    description="Words in this article",
                ),
                Property(
                    name="isAccessible",
                    data_type=DataType.BOOL,
                    description="Whether the article is currently accessible through the url",
                ),
            ],
        )

        # Author collection
        collections["Author"] = client.collections.create(
            name="Author",
            description="An author",
            vector_config=self.vector_config,
            generative_config=self._generative_config,  # Fixed: use _generative_config
            properties=[
                Property(
                    name="name",
                    data_type=DataType.TEXT,
                    description="Name of the author",
                    tokenization=Tokenization.FIELD,
                ),
            ],
        )

        # Publication collection
        collections["Publication"] = client.collections.create(
            name="Publication",
            description="A publication with an online source",
            vector_config=self.vector_config,
            generative_config=self._generative_config,  # Fixed: use _generative_config
            properties=[
                Property(
                    name="name",
                    data_type=DataType.TEXT,
                    description="Name of the publication",
                    tokenization=Tokenization.WHITESPACE,
                ),
                Property(
                    name="headquartersGeoLocation",
                    data_type=DataType.GEO_COORDINATES,
                    description="Geo location of the HQ",
                ),
            ],
        )

        # Now add reference properties

        # Add references to Article
        article_collection = client.collections.get("Article")
        article_collection.config.add_reference(
            ReferenceProperty(
                name="hasAuthors",
                target_collection="Author",
                description="Authors this article has",
            )
        )
        article_collection.config.add_reference(
            ReferenceProperty(
                name="inPublication",
                target_collection="Publication",
                description="Publication this article appeared in",
            )
        )
        article_collection.config.add_reference(
            ReferenceProperty(
                name="ofCategory",
                target_collection="Category",
                description="Category that the article belongs to",
            )
        )

        # Add references to Author
        author_collection = client.collections.get("Author")
        author_collection.config.add_reference(
            ReferenceProperty(
                name="wroteArticles",
                target_collection="Article",
                description="Articles this author wrote",
            )
        )
        author_collection.config.add_reference(
            ReferenceProperty(
                name="writesFor",
                target_collection="Publication",
                description="A publication this author has written for",
            )
        )

        # Add references to Publication
        publication_collection = client.collections.get("Publication")
        publication_collection.config.add_reference(
            ReferenceProperty(
                name="hasArticles",
                target_collection="Article",
                description="The articles this publication has",
            )
        )

        return collections

    def _get_sub_filelist(self, filedir):
        return [f for f in os.listdir(filedir) if f.endswith(".json")]

    def _load_publication_and_category(
        self, client: WeaviateClient, batch_size: int = 100
    ):
        for ctype in ["categories", "publications"]:
            datafiles = self._get_sub_filelist(os.path.join(self._datadir, ctype))
            embeddings_file = self._embeddings_files[ctype]

            with open(os.path.join(self._basedir, embeddings_file), "r") as f:
                embeddings = json.load(f)

            collection_name = "Category" if ctype == "categories" else "Publication"
            collection = client.collections.get(collection_name)

            with collection.batch.fixed_size(batch_size=batch_size) as batch:
                for dfile in datafiles:
                    with open(os.path.join(self._datadir, ctype, dfile), "r") as f:
                        data = json.load(f)
                    batch.add_object(
                        properties=data["schema"],
                        uuid=data["id"],
                        vector=embeddings[data["id"]],
                    )

    def _clean_up_newsdata(self, class_name: str, value: str) -> str:
        """
        Clean up the data.
        """
        if class_name == "Author":
            value = value.replace(" Wsj.Com", "")
            value = value.replace(".", " ")
        elif class_name == "Summary":
            value = value.replace("\n", " ")
        return value

    def _generate_uuid(self, key: str) -> str:
        """
        Generates a universally unique identifier (uuid).
        """
        return str(uuid.uuid3(uuid.NAMESPACE_DNS, key))

    def _load_authors_article(self, client: WeaviateClient, batch_size: int = 50):
        datafiles = self._get_sub_filelist(os.path.join(self._datadir))
        embedding_dict = {}

        # Load embeddings for articles and authors
        for ctype in ["articles", "authors"]:
            embeddings_file = self._embeddings_files[ctype]
            with open(os.path.join(self._basedir, embeddings_file), "r") as f:
                embedding_dict[ctype] = json.load(f)

        # Get collections
        article_collection = client.collections.get("Article")
        author_collection = client.collections.get("Author")
        publication_collection = client.collections.get("Publication")

        with article_collection.batch.fixed_size(
            batch_size=batch_size
        ) as article_batch, author_collection.batch.fixed_size(
            batch_size=batch_size
        ) as author_batch:

            for datafile in datafiles:
                try:
                    with open(os.path.join(self._datadir, datafile), "r") as f:
                        data = json.load(f)

                    article_id = self._generate_uuid(data["url"])

                    # ADD AUTHORS
                    author_ids = []
                    for author in data["authors"]:
                        if len(author.split(" ")) == 2:
                            author = self._clean_up_newsdata("Author", author)
                            author_id = self._generate_uuid(author)

                            if author_id in embedding_dict["authors"]:
                                author_batch.add_object(
                                    properties={"name": author},
                                    uuid=author_id,
                                    vector=embedding_dict["authors"][author_id],
                                )

                                author_ids.append(author_id)

                                # Add references
                                author_batch.add_reference(
                                    from_uuid=author_id,
                                    from_property="writesFor",
                                    to=data["publicationId"],
                                )

                                author_batch.add_reference(
                                    from_uuid=author_id,
                                    from_property="wroteArticles",
                                    to=article_id,
                                )
                        else:
                            author_id = data["publicationId"]
                            author_ids.append(data["publicationId"])

                    # ADD ARTICLES
                    word_count = len(" ".join(data["paragraphs"]).split(" "))
                    article_object = {
                        "title": data["title"],
                        "summary": self._clean_up_newsdata("Summary", data["summary"]),
                        "wordCount": word_count,
                        "url": data["url"],
                    }

                    # Set publication date
                    if data["pubDate"] is not None and data["pubDate"] != "":
                        article_object["publicationDate"] = data["pubDate"]

                    # Add article to weaviate
                    article_batch.add_object(
                        properties=article_object,
                        uuid=article_id,
                        vector=embedding_dict["articles"][article_id],
                    )

                    # Add references
                    article_batch.add_reference(
                        from_uuid=article_id,
                        from_property="inPublication",
                        to=data["publicationId"],
                    )

                    # Add references for authors
                    for author_id in author_ids:
                        article_batch.add_reference(
                            from_uuid=article_id,
                            from_property="hasAuthors",
                            to=author_id,
                        )

                    # Add category reference if it exists
                    if "categoryId" in data and data["categoryId"]:
                        article_batch.add_reference(
                            from_uuid=article_id,
                            from_property="ofCategory",
                            to=data["categoryId"],
                        )

                except Exception as e:
                    logging.warning(f"Error while loading {datafile}: {e}")

        # Handle publication hasArticles references in a separate batch
        publication_batch_size = min(
            batch_size, 50
        )  # Smaller batch size for cross-references
        with publication_collection.batch.fixed_size(
            batch_size=publication_batch_size
        ) as pub_batch:
            for datafile in datafiles:
                try:
                    with open(os.path.join(self._datadir, datafile), "r") as f:
                        data = json.load(f)

                    article_id = self._generate_uuid(data["url"])
                    publication_id = data["publicationId"]

                    pub_batch.add_reference(
                        from_uuid=publication_id,
                        from_property="hasArticles",
                        to=article_id,
                    )
                except Exception as e:
                    logging.warning(f"Error adding publication reference: {e}")

    def upload_dataset(
        self, client: WeaviateClient, batch_size=100, overwrite=False, compress=False
    ) -> bool:
        """
        Add the collections to the schema and upload the objects.
        """
        # Delete existing collections if overwrite=True
        if overwrite:
            for collection_name in ["Article", "Author", "Publication", "Category"]:
                try:
                    client.collections.delete(collection_name)
                except:
                    pass  # Collection might not exist

        # Configure vector index based on compression setting
        from weaviate.collections.classes.config import _VectorConfigCreate
        if not isinstance(self.vector_config, _VectorConfigCreate):
            # vector_config is a list
            if compress:
                self.vector_config[0].vector_index_config = Configure.VectorIndex.hnsw(
                    quantizer=Configure.VectorIndex.Quantizer.bq()
                )
            else:
                self.vector_config[0].vector_index_config = Configure.VectorIndex.hnsw()
        else:
            # vector_config is a single object
            if compress:
                self.vector_config.vectorIndexConfig = Configure.VectorIndex.hnsw(
                    quantizer=Configure.VectorIndex.Quantizer.bq()
                )
            else:
                self.vector_config.vectorIndexConfig = Configure.VectorIndex.hnsw()

        # Add collections to the schema
        _ = self.add_collections(client)

        # Load data
        self._load_publication_and_category(client, batch_size)
        self._load_authors_article(client, batch_size)

        return True

    def get_sample(self) -> dict:
        """
        Return a sample Article object.
        """
        datafiles = self._get_sub_filelist(os.path.join(self._datadir))
        for datafile in datafiles:
            try:
                with open(os.path.join(self._datadir, datafile), "r") as f:
                    data = json.load(f)

                word_count = len(" ".join(data["paragraphs"]).split(" "))
                article_object = {
                    "title": data["title"],
                    "summary": self._clean_up_newsdata("Summary", data["summary"]),
                    "wordCount": word_count,
                    "url": data["url"],
                }

                if data["pubDate"] is not None and data["pubDate"] != "":
                    article_object["publicationDate"] = data["pubDate"]

                return article_object
            except:
                continue

        return {"title": "Sample Article", "summary": "No actual data found"}
