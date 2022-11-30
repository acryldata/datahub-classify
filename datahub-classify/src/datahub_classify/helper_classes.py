from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InfotypeProposal:
    infotype: str
    confidence_level: float
    debug_info: Dict[str, Any]


@dataclass
class ColumnMetadata:
    meta_info: dict
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
    parent_columns: List = field(default_factory=list)
    values: List = field(default_factory=list)
    infotype_proposals: Optional[List[InfotypeProposal]] = None


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
