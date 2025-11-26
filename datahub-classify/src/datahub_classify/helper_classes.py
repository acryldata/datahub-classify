from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DebugInfo:
    name: Optional[float] = None
    description: Optional[float] = None
    datatype: Optional[float] = None
    values: Optional[float] = None


@dataclass
class InfotypeProposal:
    infotype: str
    confidence_level: float
    debug_info: DebugInfo


@dataclass
class Metadata:
    meta_info: Dict[str, Any]
    name: str = field(init=False)
    description: str = field(init=False)
    datatype: str = field(init=False)
    dataset_name: str = field(init=False)

    def __post_init__(self):
        assert "Name" in self.meta_info, "meta_info must contain 'Name' field"
        assert (
            "Description" in self.meta_info
        ), "meta_info must contain 'Description' field"
        assert "Datatype" in self.meta_info, "meta_info must contain 'Datatype' field"
        assert (
            "Dataset_Name" in self.meta_info
        ), "meta_info must contain 'Dataset_Name' field"

        self.name = self.meta_info["Name"]
        self.description = self.meta_info["Description"]
        self.datatype = self.meta_info["Datatype"]
        self.dataset_name = self.meta_info["Dataset_Name"]


@dataclass
class ColumnInfo:
    metadata: Metadata
    values: List[Any]
    infotype_proposals: Optional[List[InfotypeProposal]] = None
