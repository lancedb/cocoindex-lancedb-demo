import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

import lancedb
import ollama
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from main import IMAGES_DIR, LANCEDB_URI, TABLE_NAME, load_clip_model


class SearchRequest(BaseModel):
    query: str
    mode: Literal["text", "image"] = "text"
    limit: int = 8


def embed_text_ollama(prompt: str) -> list[float]:
    """
    Call Ollama embeddings client for nomic-embed-text.
    """
    host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_URL")
    client = ollama.Client(host=host) if host else ollama
    resp = client.embeddings(model="nomic-embed-text", prompt=prompt)
    embedding = resp.get("embedding")
    if not embedding:
        raise RuntimeError("No embedding returned from Ollama")
    return embedding


def embed_text_clip(prompt: str) -> list[float]:
    """
    Use the CLIP text encoder so text queries can retrieve similar images.
    """
    model, processor, device = load_clip_model()
    inputs = processor(  # type: ignore[call-arg]
        text=[prompt], return_tensors="pt", padding=True, truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        features = model.get_text_features(**inputs)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features[0].cpu().tolist()


def build_image_url(image_path: str | None, image_name: str | None) -> str | None:
    candidates: list[str] = []
    if image_path:
        candidates.append(Path(image_path).name)
    if image_name:
        name = Path(image_name).name
        candidates.append(name)
        if not Path(name).suffix:
            stem = Path(name).stem
            candidates.extend([f"{stem}.jpg", f"{stem}.jpeg", f"{stem}.png"])

    for filename in candidates:
        if (IMAGES_DIR / filename).exists():
            return f"/img/{filename}"

    if candidates:
        return f"/img/{candidates[0]}"
    return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_dotenv()

    db_path = os.getenv("LANCEDB_URI", LANCEDB_URI)
    table_name = os.getenv("TABLE_NAME", TABLE_NAME)

    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    app.state.db = db
    app.state.table = table

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/img", StaticFiles(directory=IMAGES_DIR), name="img")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search")
async def search(request: SearchRequest) -> dict[str, object]:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    column = "instructions_vector" if request.mode == "text" else "image_vector"

    if request.mode == "text":
        try:
            query_vector = await asyncio.to_thread(embed_text_ollama, request.query)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(
                status_code=500, detail=f"Failed to embed text query: {exc}"
            ) from exc
    else:
        try:
            query_vector = embed_text_clip(request.query)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=f"Failed to embed query: {exc}") from exc

    table = app.state.table
    results = (
        table.search(query_vector, vector_column_name=column)
        .limit(max(1, min(request.limit, 8)))
        .to_list()
    )

    payload: list[dict[str, object]] = []
    for item in results:
        payload.append(
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "ingredients": item.get("ingredients"),
                "instructions": item.get("instructions"),
                "image_name": item.get("image_name"),
                "image_path": item.get("image_path"),
                "image_url": build_image_url(item.get("image_path"), item.get("image_name")),
                "is_vegetarian": item.get("is_vegetarian"),
                "has_nuts": item.get("has_nuts"),
                "has_dairy": item.get("has_dairy"),
                "has_eggs": item.get("has_eggs"),
                "category": item.get("category"),
                "score": item.get("_distance"),
            }
        )

    return {"mode": request.mode, "results": payload}
