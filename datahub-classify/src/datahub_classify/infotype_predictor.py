import importlib
import logging
from typing import Any, Dict, List, Optional

from datahub_classify.constants import EXCLUDE_NAME
from datahub_classify.helper_classes import ColumnInfo, InfotypeProposal
from datahub_classify.infotype_utils import perform_basic_checks, strip_formatting

logger = logging.getLogger(__name__)


def get_infotype_function_mapping(
    infotypes: Optional[List[str]], global_config: Dict[str, Dict]
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
            if fn_name in module_fn_dict:
                infotype_function_map[infotype] = module_fn_dict[fn_name]
            else:
                fn_name = "inspect_for_custom_infotype"
                infotype_function_map[infotype] = module_fn_dict[fn_name]
    return infotype_function_map


def predict_infotypes(
    column_infos: List[ColumnInfo],
    confidence_level_threshold: float,
    global_config: Dict[str, Dict],
    infotypes: Optional[List[str]] = None,
    minimum_values_threshold: int = 50,
) -> List[ColumnInfo]:
    infotype_function_map = get_infotype_function_mapping(infotypes, global_config)
    logger.debug(f"Total columns to be processed --> {len(column_infos)}")
    logger.debug(f"Confidence Level Threshold set to --> {confidence_level_threshold}")
    logger.debug("===========================================================")
    basic_checks_failed_columns = []
    num_cols_with_infotype_assigned = 0
    strip_exclusion_formatting = global_config.get("strip_exclusion_formatting")

    for column_info in column_infos:
        logger.debug(
            f"processing column: {column_info.metadata.name} -- dataset: {column_info.metadata.dataset_name}"
        )
        # iterate over all infotype functions
        proposal_list = []
        for infotype, infotype_fn in infotype_function_map.items():
            # get the configuration
            config_dict = global_config[infotype]

            # convert exclude_name list into a set for o(1) checking
            if EXCLUDE_NAME in config_dict and config_dict[EXCLUDE_NAME] is not None:
                config_dict[EXCLUDE_NAME] = (
                    set(config_dict[EXCLUDE_NAME])
                    if not strip_exclusion_formatting
                    else set([strip_formatting(s) for s in config_dict[EXCLUDE_NAME]])
                )
            else:
                config_dict[EXCLUDE_NAME] = set()

            # call the infotype prediction function
            column_info.values = [
                val
                for val in column_info.values
                if str(val).strip() not in ["nan", "", "None"]
            ]
            try:
                if perform_basic_checks(
                    column_info.metadata,
                    column_info.values,
                    config_dict,
                    infotype,
                    minimum_values_threshold,
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
                    basic_checks_failed_columns.append(
                        (column_info.metadata.name, column_info.metadata.dataset_name)
                    )

            except Exception as e:
                # traceback.print_exc()
                logger.warning(f"Failed to extract info type {infotype} due to {e}")
        if len(proposal_list) > 0:
            num_cols_with_infotype_assigned += 1
        column_info.infotype_proposals = proposal_list
    if len(basic_checks_failed_columns) > 0:
        basic_checks_failed_columns_set = set(basic_checks_failed_columns)
        logger.warning(
            f"Infotype not extracted due to basic checks failure for {len(basic_checks_failed_columns_set)} out of {len(column_infos)} columns.Check DEBUG logs for details"
        )
        logger.debug(
            f"Basic Checks failed for following (column_name, table_name) --> {basic_checks_failed_columns_set}"
        )
    logger.debug(
        f"Infotype assigned to {num_cols_with_infotype_assigned} out of {len(column_infos)} columns"
    )

    return column_infos
