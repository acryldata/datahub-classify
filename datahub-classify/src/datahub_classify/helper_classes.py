from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy
from pydantic import BaseModel, Field


@dataclass
class ScoreInfo:
    confidence: Optional[float] = field(default=None)
    weighted_score: Optional[float] = field(default=None)


class DebugInfo(BaseModel):
    name: Optional[float] = Field(
        default=None, description="confidence score using name"
    )
    description: Optional[float] = Field(
        default=None, description="confidence score using description"
    )
    datatype: Optional[float] = Field(
        default=None,
        description="confidence score using datatype. For tables, it is None",
    )
    values: Optional[float] = Field(
        default=None,
        description="confidence score using values. For tables, it is None",
    )
    platform: Optional[float] = Field(
        default=None,
        description="confidence score using platform. For columns, it is None",
    )
    table_schema: Optional[float] = Field(
        default=None,
        description="For tables, it is confidence score using table schema. For columns, it is table_similarity_score",
    )
    lineage: Optional[float] = Field(
        default=None, description="confidence score using lineage"
    )


class SimilarityFactorScoreInfo(BaseModel):
    name: Optional[ScoreInfo] = Field(
        default=None,
        description="confidence score and weighted score contribution using name",
    )
    description: Optional[ScoreInfo] = Field(
        default=None,
        description="confidence score and weighted score contribution using description",
    )
    datatype: Optional[ScoreInfo] = Field(
        default=None,
        description="confidence score and weighted score contribution using datatype. For tables, it is None",
    )
    values: Optional[ScoreInfo] = Field(
        default=None,
        description="confidence score and weighted score contribution using values. For tables, it is None",
    )
    platform: Optional[ScoreInfo] = Field(
        default=None,
        description="confidence score and weighted score contribution using platform. For columns, it is None",
    )
    table_schema: Optional[ScoreInfo] = Field(
        default=None,
        description="For tables, it is confidence score and weighted score contribution using table schema. For columns, it is table_similarity_score",
    )
    lineage: Optional[ScoreInfo] = Field(
        default=None,
        description="confidence score and weighted score contribution using lineage",
    )


@dataclass
class InfotypeProposal:
    infotype: str
    confidence_level: float
    debug_info: DebugInfo


@dataclass
class TextEmbeddings:
    emb_type: str
    embedding: numpy.ndarray


@dataclass
class ColumnMetadata:
    name: Optional[str]
    datatype: Optional[str]
    dataset_name: Optional[str]
    column_id: Optional[str]
    description: Optional[str] = None
    name_embedding: List[TextEmbeddings] = field(default_factory=list)
    desc_embedding: List[TextEmbeddings] = field(default_factory=list)


@dataclass
class ColumnInfo:
    metadata: ColumnMetadata
    values: List[Any] = field(default_factory=list)
    infotype_proposals: Optional[List[InfotypeProposal]] = None
    parent_columns: List[str] = field(default_factory=list)


@dataclass
class TableMetadata:
    name: Optional[str]
    platform: Optional[str]
    table_id: Optional[str]
    description: Optional[str] = None
    name_embedding: List[TextEmbeddings] = field(default_factory=list)
    desc_embedding: List[TextEmbeddings] = field(default_factory=list)


@dataclass
class TableInfo:
    metadata: TableMetadata
    column_infos: List[ColumnInfo]
    parent_tables: List[str] = field(default_factory=list)


@dataclass
class SimilarityInfo:
    score: Optional[float]
    prediction_factors_scores: Optional[SimilarityFactorScoreInfo]
