# Keeping your data in LanceDB fresh with CocoIndex

![](./assets/coco-lance.jpg)

## Background

This repo contains a demo of using [CocoIndex](https://cocoindex.io/), a data transformation framework
the provides incremental processing and data lineage out-of-the-box, with [LanceDB](https://lancedb.com),
a multimodal lakehouse for AI.

The goal is to store a multimodal dataset (images + text) in LanceDB and keep it fresh with CocoIndex.

### Why use LanceDB?

One of the biggest benefits of using LanceDB over traditional databases or data lakes is this: Data
that would otherwise be scattered in separate directories (for e.g., when using Parquet, tables
tend to store pointer URLs to images/videos/large binary blobs, not the actual data itself) --
is *collocated* alongside the embeddings and metadata. This simplifies governance and distribution.

Another key distinguishing factor when using LanceDB is the ability to evolve schema and data
effortlessly -- Lance tables are "two-dimensional", meaning that they can grow both horizontally and
vertically in essentially a zero-cost manner. Say you want to use an LLM to extract new features
from one of the columns in a Lance table: you would run your pipeline, update the table schema to
add a new column, and backfill it with the required values by running the transform.

In traditional data lakes (e.g., based on Iceberg), this would require a full table rewrite, but in
Lance, only the new data is being written (no table locks while write happen). This means that large
teams working on multiple feature engineering tasks can simultaneously write new columns without
affecting the layout on disk.

There are several more benefits to LanceDB that leverage all the benefits of the
[Lance format](https://lance.org/), an open lakehouse format for multimodal AI, so we won't list
them all here :).

### Why do incremental processing via CocoIndex?

Not all vector processing workloads are offline batch workloads. Consider this scenario: you have a
user-facing application where users enter their recipes (along with images of the food/drink item that
they prepared), and you want to persist the data to a multimodal storage engine.
In this scenario, you typically don't begin with huge amounts of data. You accumulate
data over time, as users add their creations. And the volume/velocity of the data aren't staggeringly
high -- at times, there's ony a trickle of data coming in, but at other times, you may observe
larger volumes coming in at a higher velocity than normal.

For scenarios like this, incremental processing is an efficient technique that processes only new
or changed data (deltas) since the last update, rather than reprocessing entire large datasets. This
tends to reduce computation while lowering costs for near real-time
analytics. CocoIndex is ideal for managing constantly evolving data sources, handling small batches
of updates to keep data fresh with less overhead than full batch workloads.

[CocoIndex](https://cocoindex.io/docs/) uses a declarative approach to
defining indexing "flows", which involves source data and
transformed data (either as an intermediate result or the final result to be put into targets).
All data within the indexing flow has schema determined at flow definition time.

## Dataset

We'll be using the [food ingredients and recipes](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images)
dataset from Kaggle. The data contains 13k+ recipes and images of food/drinks scraped from the
Epicurious website. The dataset is multimodal, containing images, arrays and text.

Download the dataset from Kaggle to the local directory (it will be in a file named `archive.zip`).
Unzip the file at the root level of this repository.

## Setup

We'll use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the dependencies for
this project. Run the following command to install the required Python libraries to get started.

```bash
uv sync
```

## Generate data

To simulate a scenario where we have data intermittently coming in from a source, we'll use the
script `data_generator.py`. This script looks at the source data in the `archive` directory
and writes JSON records of the source data. The JSON records also contain a path to the image
file for the corresponding recipe ID, so that it can be easily located for ingestion into LanceDB.

```bash
uv run data_generator.py --start 0 --end 5
```

This writes out the first 5 recipe records to a JSON file in the path `data/*.json`. Simultaneously,
it also copies the image file into the `data/images/*.jpg` path.

To generate the data for the next 5 records, the corresponding start and end indices can be entered.

```bash
uv run data_generator.py --start 5 --end 10
```

To delete existing records and start afresh, use the `--refresh` flag.

```bash
uv run data_generator.py --start 0 --end 10 --refresh
```

Running the script multiple times will generate multiple JSON files, one for each run. This mimics
"real" data that may be coming from a push API in an application.

## Running the CocoIndex flow

CocoIndex uses a Postgres server to maintain a long-lived connection between the source and the
target. It's simple to get it running via Docker as follows:
```bash
docker compose -f <(curl -L https://raw.githubusercontent.com/cocoindex-io/cocoindex/refs/heads/main/dev/postgres.yaml) up
```

To run a one-time update in CocoIndex, use the following command:
```
cocoindex update main
```

[CocoInsight](https://cocoindex.io/docs/cocoinsight_access) (Free beta for now) can be used to view the index
generation and understand the data lineage of the pipeline in a GUI. Start a local CocoInsight server as follows:

```bash
cocoindex server -ci main
```

Open the CocoInsight UI at https://cocoindex.io/cocoinsight. You can run also run queries in the CocoInsight UI
to test that the search functionality is working as intended.

### Managing updates

CocoIndex will watch the source directory `data/` for any updates, and every time there is a change data capture
trigger in the source path (e.g., a new JSON file is added), this will trigger the CocoIndex server to run
the flow, and update the data.

### Data compaction and why it's needed

LanceDB uses Lance tables under the hood. Unlike Parquet (which uses row groups and partitions to store data on disk),]the Lance format uses _fragments_ and tracks data versions via a manifest. In incremental data processing pipelines such as those run
in CocoIndex, a lot of smaller fragments can add up over time, which can impact query latency as the data grows in size.
It's recommended to run compaction at periodic intervals (e.g., once every 7 days), or more frequently depending on the
volume/velocity of commits to the storage layer in a given period of time.

The `optimize()` method handles compaction, pruning of 

```py
# Open your lance table and run this command
table.optimize()
```
This performs the following:
- Compaction: Merges small files into larger ones
- Prune: Removes old versions of the dataset
- Index: Optimizes the indexes, adding new data to existing indexes

There is no need to compact tables too frequently, as this comes with computational overhead. LanceDB is highly
performant up to millions of rows, so you can adjust the frequency of compaction gradually, based on the needs
of your workloads.

---

## [Optional]: Running a pure LanceDB workflow

To contrast the CocoIndex "incremental way" with the traditional batch processing approach,
we provide an additional script, `ingest.py` that contains code that ingests the recipe
data into LanceDB. This step is optional (the aim of this repo is to show how to do it
using CocoIndex using the defined above).

The ingestion script also generates two kinds of embeddings:
- Text embeddings on the `instructions` column (TODO: concatenate the `title` and `instructions` and embed _that_ instead)
- Image embeddings on the `image` binary column

The text embeddings use the `nomic-embed-text` model via Ollama, and the image embeddings use
the `openai/clip-vit-base-patch32` model, from Hugging Face.

Run the script as follows:

```bash
# Overwrite the existing database
uv run ingest.py -o
# Or, append to an existing database (default mode)
uv run ingest.py
```

An upsert pipeline is used during ingestion, so that duplicate data isn't written to the table.
This means that as the script is run multiple times (as new data comes in), only records that have
a new unique `id` field for the recipe are written to the table.

## Querying the database

The `query.py` script contains sample code to query the data once it's persisted to LanceDB.

```bash
uv run query.py
```

Two kinds of queries are run:
- Query via a text embedding on the `instruction_vector` column
- Query via a text-to-image embedding on the `image_vector` column

Each should return relevant `top-k` results based on the query.

## Inspect the database

To inspect the table row count and list the indexes and schema of the tables
in LanceDB, run the `inspect_db.py` script.

```bash
uv run inspect_db.py
```

This can be used to track the table evolution over time.