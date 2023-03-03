## UNOFFICIAL Weaviate demo data uploader

This is an educational project that aims to make it easy to upload demo data to your instance of Weaviate. The intended use case for users learning how to use Weaviate. 

## Usage

All datasets are based on `Dataset` superclass, and includes a number of built-in methods to make it easier to work with it. 

Once you instantiate a dataset, to upload it to Weaviate the syntax is as follows:

```python
import wv_datasets
dataset = wv_datasets.JeopardyQuestionsSmall()  # Instantiate dataset
dataset.upload_dataset(client)  # Add class to schema & Upload objects (uses batch uploads by default)
```

Where `client` is the instantiated `weaviate.Client` object.

```python
import weaviate
import os
import json

wv_url = "https://some-endpoint.weaviate.network"
api_key = os.environ.get("OPENAI_API_KEY")

auth = weaviate.AuthClientPassword(
    username=os.environ.get("WCS_USER"),
    password=os.environ.get("WCS_PASS"),
)

client = weaviate.Client(
    url=wv_url,
    auth_client_secret=auth,
    additional_headers={"X-OpenAI-Api-Key": api_key},
)
```

### Built-in methods

- `.add_to_schema(client)` - add defined classes to schema; returns status & any classes already present
- `.upload_objects(client, batch_size)` - adds objects; must specify batch size
- `.upload_dataset(client)` - runs `.add_to_schema` and `.upload_objects`; default batch size 100

- `.get_class_definitions()`: See the schema definition to be added
- `.get_class_names()`: See class names in the dataset
- `.classes_in_schema(client)`: Check whether each class is already in the Weaviate schema
- `.delete_existing_dataset_classes(client)`: If dataset classes are already in the Weaviate instance, delete them from the Weaviate instance.
- `.set_vectorizer(vectorizer_name, module_config)`: Set the vectorizer and corresponding module configuration for the dataset. Datasets come pre-configured with a vectorizer & module configuration. 


## Available classes

- WikiArticles 
- WineReviews
- JeopardyQuestions1k
- JeopardyQuestions10k

## Source code:

https://github.com/databyjp/wv_demo_uploader
