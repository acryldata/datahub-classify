from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, List, Optional

import numpy
from pydantic import BaseModel, Field


@dataclass
class FactorDebugInfo:
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


class SimilarityDebugInfo(BaseModel):
    name: Optional[FactorDebugInfo] = Field(
        default=None,
        description="confidence score and weighted score contribution using name",
    )
    description: Optional[FactorDebugInfo] = Field(
        default=None,
        description="confidence score and weighted score contribution using description",
    )
    datatype: Optional[FactorDebugInfo] = Field(
        default=None,
        description="confidence score and weighted score contribution using datatype. For tables, it is None",
    )
    values: Optional[FactorDebugInfo] = Field(
        default=None,
        description="confidence score and weighted score contribution using values. For tables, it is None",
    )
    platform: Optional[FactorDebugInfo] = Field(
        default=None,
        description="confidence score and weighted score contribution using platform. For columns, it is None",
    )
    table_schema: Optional[FactorDebugInfo] = Field(
        default=None,
        description="For tables, it is confidence score and weighted score contribution using table schema. For columns, it is table_similarity_score",
    )
    lineage: Optional[FactorDebugInfo] = Field(
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
    meta_info: InitVar[Optional[Dict[str, Any]]] = None
    name: Optional[str] = None
    description: Optional[str] = None
    datatype: Optional[str] = None
    dataset_name: Optional[str] = None
    column_id: Optional[str] = None
    name_embedding: List[TextEmbeddings] = field(default_factory=list)
    desc_embedding: List[TextEmbeddings] = field(default_factory=list)

    def __post_init__(self, meta_info):
        if meta_info is not None:
            self.name = meta_info.get("Name", None)
            self.description = meta_info.get("Description", None)
            self.datatype = meta_info.get("Datatype", None)
            self.dataset_name = meta_info.get("Dataset_Name", None)
            self.column_id = meta_info.get("Column_Id", None)
            self.name_embedding = meta_info.get("name_embedding", [])
            self.desc_embedding = meta_info.get("desc_embedding", [])


@dataclass
class ColumnInfo:
    metadata: ColumnMetadata
    values: List = field(default_factory=list)
    infotype_proposals: Optional[List[InfotypeProposal]] = None
    parent_columns: List = field(default_factory=list)


@dataclass
class TableMetadata:
    meta_info: InitVar[Optional[Dict[str, Any]]] = None
    name: Optional[str] = None
    description: Optional[str] = None
    platform: Optional[str] = None
    table_id: Optional[str] = None
    name_embedding: List[TextEmbeddings] = field(default_factory=list)
    desc_embedding: List[TextEmbeddings] = field(default_factory=list)

    def __post_init__(self, meta_info):
        self.name = meta_info.get("Name", None)
        self.description = meta_info.get("Description", None)
        self.platform = meta_info.get("Platform", None)
        self.table_id = meta_info.get("Table_Id", None)
        self.name_embedding = meta_info.get("name_embedding", [])
        self.desc_embedding = meta_info.get("desc_embedding", [])


@dataclass
class TableInfo:
    metadata: TableMetadata
    column_infos: List
    parent_tables: List = field(default_factory=list)


@dataclass
class SimilarityInfo:
    score: Optional[float]
    prediction_factors_scores: Optional[SimilarityDebugInfo]
