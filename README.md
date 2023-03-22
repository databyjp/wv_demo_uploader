## UNOFFICIAL Weaviate demo data uploader

This is an educational project that aims to make it easy to upload demo data to your instance of [Weaviate](https://weaviate.io). The target audience is developers learning how to use Weaviate.

## Usage

```shell
pip install weaviate-demo-datasets
```

All datasets are based on the `Dataset` superclass, which includes a number of built-in methods to make it easier to work with it.

Each dataset includes a default vectorizer configuration for convenience, which can be:
- viewed via the `.get_class_definitions` method and
- changed via the `.set_vectorizer` method.
The target Weaviate instance must include the specified vectorizer module.

Once you instantiate a dataset, you can upload it to Weaviate with the following:

```python
import weaviate_datasets
dataset = weaviate_datasets.JeopardyQuestions10k()  # Instantiate dataset
dataset.upload_dataset(client)  # Add class to schema & upload objects (uses batch uploads by default)
```

Where `client` is the instantiated `weaviate.Client` object, such as:

```python
import weaviate
import os
import json

wv_url = "https://some-endpoint.weaviate.network"
api_key = os.environ.get("OPENAI_API_KEY")

# If authentication required (e.g. using WCS)
auth = weaviate.AuthClientPassword(
    username=os.environ.get("WCS_USER"),
    password=os.environ.get("WCS_PASS"),
)

client = weaviate.Client(
    url=wv_url,
    auth_client_secret=auth,  # If authentication required
    additional_headers={"X-OpenAI-Api-Key": api_key},  # If using OpenAI inference
)
```

### Built-in methods

- `.add_to_schema(client)` - add defined classes to schema; returns status & any classes already present
- `.upload_objects(client, batch_size)` - adds objects; must specify batch size
- `.upload_dataset(client)` - runs `.add_to_schema` and `.upload_objects`; default batch size 100

- `.get_class_definitions()`: See the schema definition to be added
- `.get_class_names()`: See class names in the dataset
- `.get_sample()`: See a sample data object
- `.classes_in_schema(client)`: Check whether each class is already in the Weaviate schema
- `.delete_existing_dataset_classes(client)`: If dataset classes are already in the Weaviate instance, delete them from the Weaviate instance.
- `.set_vectorizer(vectorizer_name, module_config)`: Set the vectorizer and corresponding module configuration for the dataset. Datasets come pre-configured with a vectorizer & module configuration.

## Available classes

### Not including vectors
- WikiArticles (A handful of Wikipedia summaries)
- WineReviews (50 wine reviews)

### Including vectors
- WikiCities (500 large cities + Wikipedia summaries, vectorized with OpenAI `text-embedding-ada-002`)
- JeopardyQuestions1k (1,000 Jeopardy questions & answers, vectorized with OpenAI `text-embedding-ada-002`)
- JeopardyQuestions10k (10,000 Jeopardy questions & answers, vectorized with OpenAI `text-embedding-ada-002`)

## Source code

https://github.com/databyjp/wv_demo_uploader
