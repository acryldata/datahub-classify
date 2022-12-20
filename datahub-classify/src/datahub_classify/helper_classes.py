from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict, Union


@dataclass
class InfotypeProposal:
    infotype: str
    confidence_level: float
    debug_info: Dict[str, Any]


@dataclass
class Metadata:
    meta_info: Dict[str, Any]
    name: str = field(init=False)
    description: str = field(init=False)
    datatype: str = field(init=False)
    dataset_name: str = field(init=False)

    def __post_init__(self):
        self.name = self.meta_info.get("Name", None)
        self.description = self.meta_info.get("Description", None)
        self.datatype = self.meta_info.get("Datatype", None)
        self.dataset_name = self.meta_info.get("Dataset_Name", None)


@dataclass
class ColumnInfo:
    metadata: Metadata
    values: List[Any]
    infotype_proposals: Optional[List[InfotypeProposal]] = None


class DebugInfo(TypedDict, total=False):
    Name: Union[str, float]
    Description: Union[str, float]
    Datatype: Union[str, float]
    Values: Union[str, float]
