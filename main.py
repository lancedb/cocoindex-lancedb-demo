import datetime
import functools
import io
import os
from pathlib import Path
from typing import Literal, cast

import cocoindex
import cocoindex.targets.lancedb as coco_lancedb
import torch
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

load_dotenv()

# Constants
DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
LANCEDB_URI = "./recipe_lancedb"
TABLE_NAME = "recipes"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TEXT_MODEL_NAME = "nomic-embed-text"
TEXT_EMBED_DIM = 768
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"


class RecipeInput(BaseModel):
    id: int
    title: str | None = None
    ingredients: list[str] | None = None
    instructions: str | None = None
    image_name: str | None = None
    image_path: str | None = None


@functools.cache
def load_clip_model() -> tuple[CLIPModel, CLIPProcessor, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CLIPModel.from_pretrained(IMAGE_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(IMAGE_MODEL_NAME)
    model.to(device)  # type: ignore[call-arg]
    model.eval()
    return model, processor, device

# --- Transform flows are transforms that are common to both indexing and querying in CocoIndex---

@cocoindex.transform_flow()
def text_to_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[list[float]]:
    return text.transform(
        cocoindex.functions.EmbedText(
            api_type=cocoindex.LlmApiType.OLLAMA,
            model=TEXT_MODEL_NAME,
            address=OLLAMA_URL,
            expected_output_dimension=TEXT_EMBED_DIM,
        )
    )

# --- CocoIndex custom functions used in flow definitions ---

@cocoindex.op.function()
def coerce_recipes(payload: cocoindex.Json) -> list[RecipeInput]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = [payload]
    else:
        return []
    return [RecipeInput.model_validate(item) for item in items]


@cocoindex.op.function()
def load_image_bytes(image_path: str | None, image_name: str | None) -> bytes | None:
    # We may or may not have images for each item
    if image_path:
        candidate = Path(image_path)
        if candidate.exists():
            return candidate.read_bytes()

    if image_name:
        name = Path(image_name).name
        candidate = IMAGES_DIR / name
        if candidate.exists():
            return candidate.read_bytes()
        # Making the assumption that image could be in one of three formats
        stem = Path(image_name).stem
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = IMAGES_DIR / f"{stem}{ext}"
            if candidate.exists():
                return candidate.read_bytes()

    return None


@cocoindex.op.function()
def concat_text(title: str | None, instructions: str | None) -> str:
    """
    We generate text embeddings over both title and instructions combined.
    """
    parts: list[str] = []
    if title:
        parts.append(title.strip())
    if instructions:
        parts.append(instructions.strip())
    return "\n\n".join(parts)


@cocoindex.op.function(cache=True, behavior_version=1, gpu=True)
def image_embedding_clip(
    image_bytes: bytes | None,
) -> cocoindex.Vector[cocoindex.Float32, Literal[512]] | None:
    """
    Generate image embedding using CLIP model
    """
    if image_bytes is None:
        return None

    model, processor, device = load_clip_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")  # type: ignore[call-arg]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        features = model.get_image_features(**inputs)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    return cast(
        cocoindex.Vector[cocoindex.Float32, Literal[512]],
        features[0].cpu().tolist(),
    )

# --- CocoIndex flow definition ---

@cocoindex.flow_def(name="RecipeIngest")
def recipe_ingest_flow(
    flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
) -> None:
    data_scope["recipe_files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path=DATA_DIR.as_posix(),
            included_patterns=["recipes_*.json"],
        ),
        refresh_interval=datetime.timedelta(seconds=5),
    )

    recipe_embeddings = data_scope.add_collector()

    with data_scope["recipe_files"].row() as doc:
        # LocalFile rows expose file metadata plus the file body in doc["content"].
        # https://cocoindex.io/docs/sources/localfile
        doc["recipes"] = (
            doc["content"]
            .transform(cocoindex.functions.ParseJson(), language="json")
            .transform(coerce_recipes)
        )

        with doc["recipes"].row() as recipe:
            recipe["image"] = recipe["image_path"].transform(
                load_image_bytes, image_name=recipe["image_name"]
            )
            recipe["text_for_embedding"] = recipe["title"].transform(
                concat_text, instructions=recipe["instructions"]
            )
            recipe["instructions_vector"] = text_to_embedding(recipe["text_for_embedding"])
            recipe["image_vector"] = recipe["image"].transform(image_embedding_clip)

            recipe_embeddings.collect(
                id=recipe["id"],
                title=recipe["title"],
                ingredients=recipe["ingredients"],
                instructions=recipe["instructions"],
                image_name=recipe["image_name"],
                image_path=recipe["image_path"],
                image=recipe["image"],
                instructions_vector=recipe["instructions_vector"],
                image_vector=recipe["image_vector"],
            )

    recipe_embeddings.export(
        "recipes",
        coco_lancedb.LanceDB(db_uri=LANCEDB_URI, table_name=TABLE_NAME),
        primary_key_fields=["id"],
    )


# --- CocoIndex query handler (optional used for running test queries downstream of the flow) ---

@recipe_ingest_flow.query_handler(
    result_fields=cocoindex.QueryHandlerResultFields(
        embedding=["instructions_vector"],
        score="score",
    ),
)
async def search(query: str) -> cocoindex.QueryOutput:
    db = await coco_lancedb.connect_async(LANCEDB_URI)
    table = await db.open_table(TABLE_NAME)

    query_embedding = await text_to_embedding.eval_async(query)
    search = await table.search(query_embedding, vector_column_name="instructions_vector")
    search_results = await search.limit(5).to_list()

    return cocoindex.QueryOutput(
        results=[
            {
                "id": result["id"],
                "title": result["title"],
                "ingredients": result["ingredients"],
                "instructions": result["instructions"],
                "image_name": result["image_name"],
                "image_path": result["image_path"],
                "score": result["_distance"],
            }
            for result in search_results
        ],
        query_info=cocoindex.QueryInfo(
            embedding=query_embedding,
            similarity_metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
        ),
    )
