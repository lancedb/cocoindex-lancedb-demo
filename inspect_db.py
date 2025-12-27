import argparse

import lancedb

LANCEDB_URI = "./recipe_lancedb"


def list_table_names(db: lancedb.db.DBConnection) -> list[str]:
    tables: list[str] = []
    page_token: str | None = None
    while True:
        response = db.list_tables(page_token=page_token)
        tables.extend(response.tables)
        if not response.page_token:
            break
        page_token = response.page_token
    return tables


def list_indexes(db_uri: str) -> None:
    db = lancedb.connect(db_uri)
    tables = list_table_names(db)
    if not tables:
        print(f"No tables found in {db_uri}.")
        return

    for table_name in tables:
        table = db.open_table(table_name)
        indexes = list(table.list_indices())
        print(f"Table \"{table_name}\" has the following indexes:")
        if not indexes:
            print("  (no indexes)")
            continue
        for index in indexes:
            columns = ", ".join(index.columns)
            print(f"  - {index.name}: {index.index_type} on [{columns}]")


def run_inspection(db_uri: str) -> None:
    db = lancedb.connect(db_uri)
    tables = list_table_names(db)

    if not tables:
        print(f"No tables found in {db_uri}.")
        return

    for table_name in tables:
        table = db.open_table(table_name)
        print(f"Table \"{table.name}\" has {table.count_rows()} rows\n---")
        print(f"Schema for \"{table_name}\" table:")
        print(table.schema, "\n---")
        list_indexes(db_uri)


def main() -> None:
    parser = argparse.ArgumentParser("List all indexes in a LanceDB database")
    parser.add_argument(
        "--db-uri",
        default=LANCEDB_URI,
        help="LanceDB URI to inspect",
    )
    args = parser.parse_args()

    run_inspection(args.db_uri)


if __name__ == "__main__":
    main()
