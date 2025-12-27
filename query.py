import lancedb
import torch

from ingest import (
    LANCEDB_URI,
    TABLE_NAME,
    embed_text,
    load_models,
)


def search_instructions(
    table: lancedb.table.Table,
    text: str,
    limit: int,
) -> None:
    query_vector = embed_text(text)
    results = (
        table.search(query_vector, vector_column_name="instructions_vector")
        .select(["id", "title", "ingredients", "image_name"])
        .limit(limit)
        .to_polars()
    )
    print("Text query results:")
    print(results)


def search_image_by_text(
    table: lancedb.table.Table,
    text: str,
    limit: int,
    processor,
    model,
    device,
) -> None:
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)  # type: ignore[call-arg]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        features = model.get_text_features(**inputs)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    query_vector = features[0].cpu().tolist()
    results = (
        table.search(query_vector, vector_column_name="image_vector")
        .select(["id", "title", "ingredients", "image_name"])
        .limit(limit)
        .to_polars()
    )
    print("Text-to-image query results:")
    print(results)


def main() -> None:
    image_processor, image_model, device = load_models()
    db = lancedb.connect(LANCEDB_URI)
    table = db.open_table(TABLE_NAME)

    LIMIT = 2

    # Search instructions by text
    TEXT_QUERY = "vegetarian stew with onions and tomatoes"
    search_instructions(table, TEXT_QUERY, LIMIT)

    # Search images by text
    IMAGE_TEXT_QUERY = "meat and bread casserole"
    search_image_by_text(
        table,
        IMAGE_TEXT_QUERY,
        LIMIT,
        image_processor,
        image_model,
        device,
    )

    print(f"LanceDB table \"{table.name}\" has {table.count_rows()} rows.")


if __name__ == "__main__":
    main()
