"""Cold storage tiering, summaries, and tombstones.

Revision ID: 0002_cold_storage
Revises: 0001_initial
Create Date: 2026-01-09
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0002_cold_storage"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    memory_tier_enum = postgresql.ENUM("hot", "cold", name="memory_tier")
    tombstone_action_enum = postgresql.ENUM(
        "archived", "rehydrated", "purged", "summarized", name="tombstone_action"
    )
    memory_tier_enum.create(bind, checkfirst=True)
    tombstone_action_enum.create(bind, checkfirst=True)

    op.alter_column(
        "observations",
        "last_accessed",
        new_column_name="last_accessed_at",
    )
    op.execute("UPDATE observations SET access_count = 0 WHERE access_count IS NULL")
    op.alter_column(
        "observations",
        "access_count",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        nullable=False,
        server_default="0",
    )
    op.add_column(
        "observations",
        sa.Column("tier", memory_tier_enum, nullable=False, server_default="hot"),
    )
    op.add_column("observations", sa.Column("archived_at", sa.DateTime(timezone=True)))
    op.add_column("observations", sa.Column("archived_reason", sa.Text()))
    op.add_column("observations", sa.Column("archived_by", sa.String(length=100)))
    op.add_column(
        "observations",
        sa.Column("score", sa.Float(), nullable=False, server_default="0.0"),
    )
    op.add_column(
        "observations",
        sa.Column("floor_score", sa.Float(), nullable=False, server_default="-9999.0"),
    )
    op.add_column(
        "observations",
        sa.Column(
            "purge_eligible",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index("ix_observations_tier", "observations", ["tier"])
    op.create_index("ix_observations_score", "observations", ["score"])

    op.alter_column(
        "patterns",
        "last_accessed",
        new_column_name="last_accessed_at",
    )
    op.execute("UPDATE patterns SET access_count = 0 WHERE access_count IS NULL")
    op.alter_column(
        "patterns",
        "access_count",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        nullable=False,
        server_default="0",
    )
    op.add_column(
        "patterns",
        sa.Column("tier", memory_tier_enum, nullable=False, server_default="hot"),
    )
    op.add_column("patterns", sa.Column("archived_at", sa.DateTime(timezone=True)))
    op.add_column("patterns", sa.Column("archived_reason", sa.Text()))
    op.add_column("patterns", sa.Column("archived_by", sa.String(length=100)))
    op.add_column(
        "patterns",
        sa.Column("score", sa.Float(), nullable=False, server_default="0.0"),
    )
    op.add_column(
        "patterns",
        sa.Column("floor_score", sa.Float(), nullable=False, server_default="-9999.0"),
    )
    op.add_column(
        "patterns",
        sa.Column(
            "purge_eligible",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index("ix_patterns_tier", "patterns", ["tier"])
    op.create_index("ix_patterns_score", "patterns", ["score"])

    op.alter_column(
        "concepts",
        "last_accessed",
        new_column_name="last_accessed_at",
    )
    op.execute("UPDATE concepts SET access_count = 0 WHERE access_count IS NULL")
    op.alter_column(
        "concepts",
        "access_count",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        nullable=False,
        server_default="0",
    )
    op.add_column(
        "concepts",
        sa.Column("tier", memory_tier_enum, nullable=False, server_default="hot"),
    )
    op.add_column("concepts", sa.Column("archived_at", sa.DateTime(timezone=True)))
    op.add_column("concepts", sa.Column("archived_reason", sa.Text()))
    op.add_column("concepts", sa.Column("archived_by", sa.String(length=100)))
    op.add_column(
        "concepts",
        sa.Column("score", sa.Float(), nullable=False, server_default="0.0"),
    )
    op.add_column(
        "concepts",
        sa.Column("floor_score", sa.Float(), nullable=False, server_default="-9999.0"),
    )
    op.add_column(
        "concepts",
        sa.Column(
            "purge_eligible",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index("ix_concepts_tier", "concepts", ["tier"])
    op.create_index("ix_concepts_score", "concepts", ["score"])

    op.alter_column(
        "documents",
        "last_accessed",
        new_column_name="last_accessed_at",
    )
    op.execute("UPDATE documents SET access_count = 0 WHERE access_count IS NULL")
    op.alter_column(
        "documents",
        "access_count",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        nullable=False,
        server_default="0",
    )
    op.add_column(
        "documents",
        sa.Column("tier", memory_tier_enum, nullable=False, server_default="hot"),
    )
    op.add_column("documents", sa.Column("archived_at", sa.DateTime(timezone=True)))
    op.add_column("documents", sa.Column("archived_reason", sa.Text()))
    op.add_column("documents", sa.Column("archived_by", sa.String(length=100)))
    op.add_column(
        "documents",
        sa.Column("score", sa.Float(), nullable=False, server_default="0.0"),
    )
    op.add_column(
        "documents",
        sa.Column("floor_score", sa.Float(), nullable=False, server_default="-9999.0"),
    )
    op.add_column(
        "documents",
        sa.Column(
            "purge_eligible",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index("ix_documents_tier", "documents", ["tier"])
    op.create_index("ix_documents_score", "documents", ["score"])

    op.create_table(
        "memory_summaries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=True),
        sa.Column("source_ids", postgresql.JSONB, nullable=True),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "access_count",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True)),
        sa.Column("tier", memory_tier_enum, nullable=False, server_default="hot"),
        sa.Column("archived_at", sa.DateTime(timezone=True)),
        sa.Column("archived_reason", sa.Text()),
        sa.Column("archived_by", sa.String(length=100)),
        sa.Column("score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("floor_score", sa.Float(), nullable=False, server_default="-9999.0"),
        sa.Column(
            "purge_eligible",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index(
        "ix_memory_summaries_source",
        "memory_summaries",
        ["source_type", "source_id"],
    )
    op.create_index("ix_memory_summaries_tier", "memory_summaries", ["tier"])
    op.create_index("ix_memory_summaries_score", "memory_summaries", ["score"])

    op.create_table(
        "memory_tombstones",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("memory_id", sa.String(length=255), nullable=False),
        sa.Column("action", tombstone_action_enum, nullable=False),
        sa.Column("from_tier", memory_tier_enum, nullable=True),
        sa.Column("to_tier", memory_tier_enum, nullable=True),
        sa.Column("reason", sa.Text()),
        sa.Column("actor", sa.String(length=100)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
    )
    op.create_index(
        "ix_memory_tombstones_memory_id",
        "memory_tombstones",
        ["memory_id"],
    )
    op.create_index(
        "ix_memory_tombstones_action",
        "memory_tombstones",
        ["action"],
    )


def downgrade() -> None:
    op.drop_index("ix_memory_tombstones_action", table_name="memory_tombstones")
    op.drop_index("ix_memory_tombstones_memory_id", table_name="memory_tombstones")
    op.drop_table("memory_tombstones")

    op.drop_index("ix_memory_summaries_score", table_name="memory_summaries")
    op.drop_index("ix_memory_summaries_tier", table_name="memory_summaries")
    op.drop_index("ix_memory_summaries_source", table_name="memory_summaries")
    op.drop_table("memory_summaries")

    op.drop_index("ix_documents_score", table_name="documents")
    op.drop_index("ix_documents_tier", table_name="documents")
    op.drop_column("documents", "purge_eligible")
    op.drop_column("documents", "floor_score")
    op.drop_column("documents", "score")
    op.drop_column("documents", "archived_by")
    op.drop_column("documents", "archived_reason")
    op.drop_column("documents", "archived_at")
    op.drop_column("documents", "tier")
    op.alter_column(
        "documents",
        "access_count",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        nullable=True,
        server_default=None,
    )
    op.alter_column(
        "documents",
        "last_accessed_at",
        new_column_name="last_accessed",
    )

    op.drop_index("ix_concepts_score", table_name="concepts")
    op.drop_index("ix_concepts_tier", table_name="concepts")
    op.drop_column("concepts", "purge_eligible")
    op.drop_column("concepts", "floor_score")
    op.drop_column("concepts", "score")
    op.drop_column("concepts", "archived_by")
    op.drop_column("concepts", "archived_reason")
    op.drop_column("concepts", "archived_at")
    op.drop_column("concepts", "tier")
    op.alter_column(
        "concepts",
        "access_count",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        nullable=True,
        server_default=None,
    )
    op.alter_column(
        "concepts",
        "last_accessed_at",
        new_column_name="last_accessed",
    )

    op.drop_index("ix_patterns_score", table_name="patterns")
    op.drop_index("ix_patterns_tier", table_name="patterns")
    op.drop_column("patterns", "purge_eligible")
    op.drop_column("patterns", "floor_score")
    op.drop_column("patterns", "score")
    op.drop_column("patterns", "archived_by")
    op.drop_column("patterns", "archived_reason")
    op.drop_column("patterns", "archived_at")
    op.drop_column("patterns", "tier")
    op.alter_column(
        "patterns",
        "access_count",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        nullable=True,
        server_default=None,
    )
    op.alter_column(
        "patterns",
        "last_accessed_at",
        new_column_name="last_accessed",
    )

    op.drop_index("ix_observations_score", table_name="observations")
    op.drop_index("ix_observations_tier", table_name="observations")
    op.drop_column("observations", "purge_eligible")
    op.drop_column("observations", "floor_score")
    op.drop_column("observations", "score")
    op.drop_column("observations", "archived_by")
    op.drop_column("observations", "archived_reason")
    op.drop_column("observations", "archived_at")
    op.drop_column("observations", "tier")
    op.alter_column(
        "observations",
        "access_count",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        nullable=True,
        server_default=None,
    )
    op.alter_column(
        "observations",
        "last_accessed_at",
        new_column_name="last_accessed",
    )

    bind = op.get_bind()
    tombstone_action_enum = postgresql.ENUM(
        "archived", "rehydrated", "purged", "summarized", name="tombstone_action"
    )
    memory_tier_enum = postgresql.ENUM("hot", "cold", name="memory_tier")
    tombstone_action_enum.drop(bind, checkfirst=True)
    memory_tier_enum.drop(bind, checkfirst=True)
