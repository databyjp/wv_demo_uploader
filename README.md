## UNOFFICIAL Weaviate demo data uploader

This is an educational project that aims to make it easy to upload demo data to your instance of [Weaviate](https://weaviate.io). The target audience is developers learning how to use Weaviate.

## Usage

```shell
pip install -U weaviate-demo-datasets
```

Each dataset includes a default vectorizer configuration for convenience.
The target Weaviate instance must include the specified vectorizer module.

Once you instantiate a dataset, you can upload it to Weaviate with the following:

```python
import weaviate_datasets as wd
dataset = wd.JeopardyQuestions1k()  # Instantiate dataset
dataset.upload_dataset(client)  # Pass the Weaviate client instance
```

Where `client` is the instantiated `weaviate.WeaviateClient` object, such as:

```python
import weaviate
import os

client = weaviate.connect_to_local(
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY")}
)
```

To use a `weaviate.Client` object, use 0.5.x or older version of this package.

### Built-in methods
- `.upload_dataset(client)` - add defined classes to schema, adds objects
- `.get_sample()` - yields sample data object(s)

## Available classes

- Wiki100 (Top 100 Wikipedia articles)
  - `WikiChunk` collection
  - Various chunking options available:
    - Default: `wiki_sections` (sections of the Wikipedia article)
    - `wiki_section_chunked` (sections of the Wikipedia article, chunked into 200 character chunks)
    - `wiki_heading_only` (only the headings of the Wikipedia article sections)
    - `fixed` (fixed length chunks of 200 characters)
  - Use it as follows:
    ```python
    d = wd.Wiki100()
    d.collection_name = "WikiChunk"
    d.set_chunking("wiki_section_chunked")
    upload_responses = d.upload_dataset(client, overwrite=True)
    ```

- WineReviews (50 wine reviews)
  - `WineReview` collection
- WineReviewsNV (50 wine reviews)
  - `WineReviewNV` collection, with named vectors ("title", "review_body", and "title_country")
    - "title_country" -> Vector from concatenation of "title" + "country"
- WineReviewsMT (50 wine reviews)
  - `WineReviewMT` collection, tenants `tenantA` and `tenantB`
- JeopardyQuestions1k (1,000 Jeopardy questions & answers, vectorized with OpenAI `text-embedding-ada-002`)
  - `JeopardyQuestion` and `JeopardyCategory` collections
- JeopardyQuestions10k (10,000 Jeopardy questions & answers, vectorized with OpenAI `text-embedding-ada-002`)
  - `JeopardyQuestion` and `JeopardyCategory` collections

## Available classes - V3 collection

These are available with a `V3` suffix, and are compatible with the Weaviate Python client `v3.x`.

#### Not including vectors
- WineReviews (50 wine reviews)
- WineReviewsMT (50 wine reviews, multi-tenancy enabled)

#### Including vectors
- JeopardyQuestions1k (1,000 Jeopardy questions & answers, vectorized with OpenAI `text-embedding-ada-002`)
- JeopardyQuestions10k (10,000 Jeopardy questions & answers, vectorized with OpenAI `text-embedding-ada-002`)
- JeopardyQuestions1kMT (1,000 Jeopardy questions & answers, multi-tenancy enabled, vectorized with OpenAI `text-embedding-ada-002`)
- NewsArticles (News articles, including their corresponding publications, authors & categories, vectorized with OpenAI `text-embedding-ada-002`)

## Data sources

https://www.kaggle.com/datasets/zynicide/wine-reviews
https://www.kaggle.com/datasets/tunguz/200000-jeopardy-questions
https://github.com/weaviate/DEMO-NewsPublications

## Source code

https://github.com/databyjp/wv_demo_uploader
