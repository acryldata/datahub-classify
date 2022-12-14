import importlib
import logging
from typing import Any, Dict, List, Optional

from datahub_classify.helper_classes import ColumnInfo, InfotypeProposal
from datahub_classify.infotype_utils import perform_basic_checks

logger = logging.getLogger(__name__)


def get_infotype_function_mapping(
    infotypes: Optional[List[str]], global_config: Dict[str, Any]
) -> Dict[str, Any]:
    from inspect import getmembers, isfunction

    module_name = "datahub_classify.infotype_helper"
    module = importlib.import_module(module_name)
    module_fn_dict = dict(getmembers(module, isfunction))
    infotype_function_map = {}
    if not infotypes:
        infotypes = list(global_config.keys())
    for infotype in infotypes:
        if infotype not in global_config.keys():
            logger.warning(f"Configuration is not available for infotype - {infotype}")
        else:
            fn_name = f"inspect_for_{infotype.lower()}"
            infotype_function_map[infotype] = module_fn_dict[fn_name]
    return infotype_function_map


def predict_infotypes(
    column_infos: List[ColumnInfo],
    confidence_level_threshold: float,
    global_config: Dict[str, Any],
    infotypes: Optional[List[str]] = None,
) -> List[ColumnInfo]:
    infotype_function_map = get_infotype_function_mapping(infotypes, global_config)
    logger.info(f"Total columns to be processed --> {len(column_infos)}")
    logger.info(f"Confidence Level Threshold set to --> {confidence_level_threshold}")
    logger.info("===========================================================")
    for column_info in column_infos:
        logger.debug(
            f"processing column: {column_info.metadata.name} -- dataset: {column_info.metadata.dataset_name}"
        )
        # iterate over all infotype functions
        proposal_list = []
        for infotype, infotype_fn in infotype_function_map.items():
            # get the configuration
            config_dict = global_config[infotype]

            # call the infotype prediction function
            column_info.values = [
                val
                for val in column_info.values
                if str(val).strip() not in ["nan", "", "None"]
            ]
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
                        f"Failed basic checks for infotype - {infotype}, column - {column_info.metadata.name} and dataset - {column_info.metadata.dataset_name}"
                    )
            except Exception as e:
                # traceback.print_exc()
                logger.warning(f"Failed to extract info type due to {e}")
        column_info.infotype_proposals = proposal_list
    logger.info("=============RUN COMPLETE ==============================")

    return column_infos
