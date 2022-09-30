class InfotypeProposal:

    def __init__(self, infotype, confidence_level, debug_info):
        self.infotype = infotype
        self.confidence_level = confidence_level
        self.debug_info = debug_info


class ColumnInfo:

    def __init__(self, metadata, values, infotype_proposals=None):
        if infotype_proposals is None:
            infotype_proposals = []
        self.metadata = metadata
        self.values = values
        self.infotype_proposals = infotype_proposals


class Metadata:

    def __init__(self, meta_info):
        self.name = meta_info.get('Name', None)
        self.description = meta_info.get('Description', None)
        self.datatype = meta_info.get('Datatype', None)
        self.dataset_name = meta_info.get('Dataset_Name', None)
