"""
Add new columns to LanceDB database
"""

import lancedb
import pyarrow as pa

# Path to existing database
LANCEDB_URI = "./recipe_lancedb"


def drop_existing_columns(table: lancedb.table.Table) -> None:
    # For quick testing / cleanup, uncomment to drop the columns that were just created:
    table.drop_columns(["is_vegetarian", "has_nuts", "has_dairy", "has_eggs", "category"])

    # Add new typed columns initialized with nulls.
    print(f'Schema for "{table.name}" table:')
    print(table.schema, "\n---")


def add_new_column(db: lancedb.db.DBConnection) -> None:
    table_name = "recipes"
    table = db.open_table(table_name)
    # Define new column with specified types
    fields = [
        pa.field("is_vegetarian", pa.bool_()),
        pa.field("has_nuts", pa.bool_()),
        pa.field("has_dairy", pa.bool_()),
        pa.field("has_eggs", pa.bool_()),
        pa.field("category", pa.string()),
    ]

    existing = set(table.schema.names)
    fields_to_add = [field for field in fields if field.name not in existing]
    if not fields_to_add:   
        print(f'No new columns to add to "{table_name}".')

    # Add new typed columns initialized with nulls.
    table.add_columns(fields_to_add)

    # # For quick testing, delete the newly created columns if required
    # drop_existing_columns(table)

    print(f'Schema for "{table.name}" table:')
    print(table.schema, "\n---")

    # Optimize table (compact files, prune deleted columns and rebuild indexes)
    table.optimize()
    print("Table optimized after adding new columns.")


def main() -> None:
    db = lancedb.connect(LANCEDB_URI)
    # Table name
    add_new_column(db)


if __name__ == "__main__":
    main()
