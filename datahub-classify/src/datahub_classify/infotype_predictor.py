import importlib
import logging
from typing import List

import pandas as pd

from datahub_classify.helper_classes import ColumnInfo, InfotypeProposal
from datahub_classify.infotype_utils import perform_basic_checks
from datahub_classify.supported_infotypes import infotypes_to_use

logger = logging.getLogger(__name__)


def get_infotype_function_mapping(infotypes):
    from inspect import getmembers, isfunction

    module_name = "datahub_classify.infotype_helper"
    module = importlib.import_module(module_name)
    module_fn_dict = dict(getmembers(module, isfunction))
    infotype_function_map = {}
    if not infotypes:
        infotypes = infotypes_to_use
    for infotype in infotypes:
        fn_name = "inspect_for_%s" % infotype.lower()
        infotype_function_map[infotype] = module_fn_dict[fn_name]
    return infotype_function_map


def predict_infotypes(
    column_infos: List[ColumnInfo],
    confidence_level_threshold: float,
    global_config: dict,
    infotypes: List[str] = None,
) -> List[ColumnInfo]:
    # assert type(column_infos) == list, "type of column_infos should be list"
    infotype_function_map = get_infotype_function_mapping(infotypes)
    logger.info(f"Total columns to be processed --> {len(column_infos)}")
    logger.info(f"Confidence Level Threshold set to --> {confidence_level_threshold}")
    logger.info("===========================================================")
    for column_info in column_infos:
        logger.debug(f"processing column: {column_info.metadata.name}")
        # iterate over all infotype functions
        proposal_list = []
        for infotype, infotype_fn in infotype_function_map.items():
            # get the configuration
            config_dict = global_config[infotype]

            # call the infotype prediction function
            column_info.values = pd.Series(column_info.values).dropna()
            try:
                if perform_basic_checks(
                    column_info.metadata, column_info.values, config_dict, infotype
                ):
                    confidence_level, debug_info = infotype_fn(
                        column_info.metadata, column_info.values, config_dict
                    )
                    if confidence_level > confidence_level_threshold:
                        infotype_proposal = InfotypeProposal(
                            infotype, confidence_level, debug_info
                        )
                        proposal_list.append(infotype_proposal)
                else:
                    raise Exception(
                        "Failed basic checks for infotype - %s and column - %s"
                        % (infotype, column_info.metadata.name)
                    )
            except Exception as e:
                # traceback.print_exc()
                logger.warning(f"Failed to extract info type due to {e}")
        column_info.infotype_proposals = proposal_list
    logger.info("=============RUN COMPLETE ==============================")

    return column_infos
