from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


@dataclass
class InfotypeProposal:
    infotype: str
    confidence_level: float
    debug_info: Dict[str, Any]


@dataclass
class ColumnMetadata:
    meta_info: Dict[str, Any]
    name: str = field(init=False)
    description: str = field(init=False)
    datatype: str = field(init=False)
    dataset_name: str = field(init=False)
    column_id: str = field(init=False)

    def __post_init__(self):
        self.name = self.meta_info.get("Name", None)
        self.description = self.meta_info.get("Description", None)
        self.datatype = self.meta_info.get("Datatype", None)
        self.dataset_name = self.meta_info.get("Dataset_Name", None)
        self.column_id = self.meta_info.get("Column_Id", None)


@dataclass
class ColumnInfo:
    metadata: ColumnMetadata
    values: List = field(default_factory=list)
    infotype_proposals: Optional[List[InfotypeProposal]] = None
    parent_columns: List = field(default_factory=list)


@dataclass
class TableMetadata:
    meta_info: dict
    name: str = field(init=False)
    description: str = field(init=False)
    platform: str = field(init=False)
    table_id: str = field(init=False)

    def __post_init__(self):
        self.name = self.meta_info.get("Name", None)
        self.description = self.meta_info.get("Description", None)
        self.platform = self.meta_info.get("Platform", None)
        self.table_id = self.meta_info.get("Table_Id", None)


@dataclass
class TableInfo:
    metadata: TableMetadata
    column_infos: List
    parent_tables: List = field(default_factory=list)


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


@dataclass
class SimilarityInfo:
    score: Optional[float]
    prediction_factor_confidence: Optional[DebugInfo]
