from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag_llm.config.model_config import ModelConfig
from graphrag_llm.config.types import LLMProviderType
from graphrag_llm.embedding import create_embedding
from graphrag_vectors.lancedb import LanceDBVectorStore

def graphrag_retriever(query: str, **build_context_kwargs: Any) -> list[str]:
    """Build GraphRAG context for a query and return the resulting context chunks.

    Parameters
    ----------
    query : str
        The user query to retrieve context for.
    **build_context_kwargs
        Extra keyword arguments forwarded to LocalSearchMixedContext.build_context().
        Common examples include max_context_tokens, text_unit_prop, community_prop,
        top_k_mapped_entities, top_k_relationships, use_community_summary,
        and min_community_rank.

    Returns
    -------
    list[str]
        The context chunks produced by GraphRAG.
    """

    output_dir = build_context_kwargs.pop("output_dir", None)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "output"
    else:
        output_dir = Path(output_dir)

    lancedb_uri = build_context_kwargs.pop("lancedb_uri", None)
    if lancedb_uri is None:
        lancedb_uri = str(output_dir / "lancedb")

    community_level = build_context_kwargs.pop("community_level", 2)
    embedding_config = build_context_kwargs.pop("embedding_config", None)

    build_context_kwargs.setdefault("max_context_tokens", 4000)
    build_context_kwargs.setdefault("top_k_mapped_entities", 1)
    build_context_kwargs.setdefault("top_k_relationships", 1)
    build_context_kwargs.setdefault("use_community_summary", True)
    build_context_kwargs.setdefault("min_community_rank", 1)

    output_dir = str(output_dir)

    entity_df = pd.read_parquet(f"{output_dir}/entities.parquet")
    community_df = pd.read_parquet(f"{output_dir}/communities.parquet")
    entities = read_indexer_entities(entity_df, community_df, community_level)

    relationship_df = pd.read_parquet(f"{output_dir}/relationships.parquet")
    relationships = read_indexer_relationships(relationship_df)

    report_df = pd.read_parquet(f"{output_dir}/community_reports.parquet")
    reports = read_indexer_reports(report_df, community_df, community_level)

    text_unit_df = pd.read_parquet(f"{output_dir}/text_units.parquet")
    text_units = read_indexer_text_units(text_unit_df)

    vector_store = LanceDBVectorStore(
        db_uri=lancedb_uri,
        index_name="entity_description",
    )
    vector_store.connect()

    if embedding_config is None:
        embedding_config = ModelConfig(
            type=LLMProviderType.LiteLLM,
            model_provider="openai",
            model="mxbai-embed",
            api_key="sk-1234567890",
            api_base="http://localhost:4000/v1",
            model_id="default_embedding_model",
        )

    llm_embedding = create_embedding(embedding_config)

    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        entity_text_embeddings=vector_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=llm_embedding,
    )

    context_result = context_builder.build_context(query=query, **build_context_kwargs)
    return context_result.context_chunks
