from supported_infotypes import infotypes_to_use
from helper_classes import InfotypeProposal
import pandas as pd


def get_infotype_function_mapping():
    from inspect import getmembers, isfunction
    module_name = 'infotype_helper'
    module = __import__(module_name)
    module_fn_dict = dict(getmembers(module, isfunction))
    infotype_function_map = {}
    for infotype in infotypes_to_use:
        fn_name = 'inspect_for_%s' % infotype.lower()
        infotype_function_map[infotype] = module_fn_dict[fn_name]
    return infotype_function_map


def predict_infotypes(column_infos, confidence_level_threshold, global_config):
    # assert type(column_infos) == list, "type of column_infos should be list"
    infotype_function_map = get_infotype_function_mapping()
    for column_info in column_infos:
        print("processing column: ", column_info.metadata.name)

        # iterate over all infotype functions
        proposal_list = []
        for infotype, infotype_fn in infotype_function_map.items():
            # get the configuration
            config_dict = global_config[infotype]

            # call the infotype prediction function
            # TODO: Raise an exception if total non-null values are less than 50
            column_info.values = pd.Series(column_info.values).dropna()
            confidence_level, debug_info = infotype_fn(column_info.metadata, column_info.values, config_dict)

            if confidence_level > confidence_level_threshold:
                infotype_proposal = InfotypeProposal(infotype, confidence_level, debug_info)
                proposal_list.append(infotype_proposal)
            column_info.infotype_proposals = proposal_list
    print("===========================================")
    return column_infos
