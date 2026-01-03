"""
Quick migration to add alias_key column to concept_aliases table
"""
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    # Drop and recreate the table (it's empty anyway)
    conn.execute(text("DROP TABLE IF EXISTS concept_aliases"))
    conn.execute(text("""
        CREATE TABLE concept_aliases (
            alias VARCHAR(255) PRIMARY KEY,
            alias_key VARCHAR(255) NOT NULL,
            concept_id INTEGER NOT NULL REFERENCES concepts(id),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """))
    conn.execute(text("CREATE INDEX ix_concept_aliases_alias_key ON concept_aliases(alias_key)"))
    conn.commit()
    print("Migration complete: concept_aliases table recreated with alias_key column")
