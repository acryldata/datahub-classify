import itertools
import json
import logging
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pytest

from datahub_classify.helper_classes import TableInfo
from datahub_classify.similarity_predictor import check_similarity

logger = logging.getLogger(__name__)

PRUNING_THRESHOLD = 0.8
FINAL_THRESHOLD = 0.6
COLUMN_SIMILARITY_THRESHOLD = 0.8
CURRENT_WDR = os.path.dirname(os.path.abspath(__file__))
TABLE_INFOS_PATH = os.path.join(CURRENT_WDR, "table_info_objects.pickle")
TABLE_INFO_COPIES_PATH = os.path.join(CURRENT_WDR, "logical_copies.pickle")
INPUT_JSONS_PATH = os.path.join(CURRENT_WDR, "expected_output")


def load_jsons(
    input_jsons_dir: str,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, float]]:
    with open(
        os.path.join(input_jsons_dir, "pruning_table_similarity_labels_EXPECTED.json")
    ) as filename:
        pruning_table_similarity_labels_expected_ = json.load(filename)
    with open(
        os.path.join(
            input_jsons_dir, "post_pruning_table_similarity_labels_EXPECTED.json"
        )
    ) as filename:
        post_pruning_table_similarity_labels_expected_ = json.load(filename)

    with open(
        os.path.join(INPUT_JSONS_PATH, "column_similarity_scores_EXPECTED.json")
    ) as filename_:
        column_similarity_scores_expected_ = json.load(filename_)

    return (
        pruning_table_similarity_labels_expected_,
        post_pruning_table_similarity_labels_expected_,
        column_similarity_scores_expected_,
    )


def get_predicted_expected_similarity_scores_mapping_for_tables(
    predicted_similarity_labels_unit_testing: Dict[str, str],
    expected_similarity_labels_unit_testing: Dict[str, str],
) -> List[Tuple[str, str, str, str]]:
    """generate mapping of predicted - expected similarity scores, required for unit testing"""

    mapping = []
    for key_ in predicted_similarity_labels_unit_testing.keys():
        pair = key_.split("_SPLITTER_", 1)
        if expected_similarity_labels_unit_testing.get(key_, None):
            expected_similarity_label_unit_testing = (
                expected_similarity_labels_unit_testing[key_]
            )
            mapping.append(
                (
                    pair[0],
                    pair[1],
                    predicted_similarity_labels_unit_testing[key_],
                    expected_similarity_label_unit_testing,
                )
            )
    return mapping


def get_predicted_expected_similarity_scores_mapping_for_columns(
    predicted_similarity_scores_unit_testing: Dict[str, float],
    expected_similarity_scores_unit_testing: Dict[str, float],
) -> List[Tuple[str, str, float, str, float, str]]:
    """generate mapping of predicted - expected similarity scores, required for unit testing"""

    mapping = []
    for pair in predicted_similarity_scores_unit_testing.keys():
        key_ = f"{pair[0]}_COLSPLITTER_{pair[1]}"
        if key_ in expected_similarity_scores_unit_testing.keys():
            if expected_similarity_scores_unit_testing[key_] is None:
                expected_similarity_label = "not_similar"
                expected_similarity_score = 0.0
            elif (
                expected_similarity_scores_unit_testing[key_]
                >= COLUMN_SIMILARITY_THRESHOLD
            ):
                expected_similarity_label = "similar"
                expected_similarity_score = expected_similarity_scores_unit_testing[
                    key_
                ]
            else:
                expected_similarity_label = "not_similar"
                expected_similarity_score = expected_similarity_scores_unit_testing[
                    key_
                ]
            if predicted_similarity_scores_unit_testing[pair] is None:
                predicted_similarity_label = "not_similar"
                predicted_similarity_score = 0.0
            elif (
                predicted_similarity_scores_unit_testing[pair]
                >= COLUMN_SIMILARITY_THRESHOLD
            ):
                predicted_similarity_label = "similar"
                predicted_similarity_score = predicted_similarity_scores_unit_testing[
                    pair
                ]
            else:
                predicted_similarity_label = "not_similar"
                predicted_similarity_score = predicted_similarity_scores_unit_testing[
                    pair
                ]
            mapping.append(
                (
                    pair[0],
                    pair[1],
                    predicted_similarity_score,
                    predicted_similarity_label,
                    expected_similarity_score,
                    expected_similarity_label,
                )
            )
    return mapping


def get_table_infos_and_pairs() -> Tuple[Dict[str, TableInfo], List[Tuple[str, str]]]:
    with open(TABLE_INFOS_PATH, "rb") as table_info_file:
        table_infos = pickle.load(table_info_file)
    if os.path.exists(TABLE_INFO_COPIES_PATH):
        with open(TABLE_INFO_COPIES_PATH, "rb") as table_info_copies_file:
            table_info_copies = pickle.load(table_info_copies_file)
    else:
        table_info_copies = None

    logger.info("Creating Table Pairs List................")
    table_pairs = list(itertools.combinations(table_infos.keys(), 2))
    if table_info_copies:
        for key in table_infos.keys():
            table_pairs.append((key, f"{key}_LOGICAL_COPY"))
        table_infos.update(table_info_copies)

    return table_infos, table_pairs


def get_similarity_predictions(
    table_infos: Dict[str, TableInfo], table_pairs: List[Tuple[str, str]]
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, float]]:
    columns_predicted_scores: Dict[str, float] = dict()
    pruning_mode_results: Dict[str, Tuple] = dict()
    post_pruning_mode_results: Dict[str, Tuple] = dict()

    logger.info("Starting check similarity in pruning mode.............")
    for table_pair in table_pairs:
        table_pair_list = sorted(table_pair, key=str.lower)
        table_pair = (table_pair_list[0], table_pair_list[1])
        pruning_mode_results[
            f"{table_pair[0]}_SPLITTER_{table_pair[1]}"
        ] = check_similarity(
            table_infos[table_pair[0]],
            table_infos[table_pair[1]],
            pruning_mode=True,
            use_embeddings=False,
        )
    pruning_mode_output_predicted = {
        key: ("not_similar" if value[0].score < PRUNING_THRESHOLD else "similar")
        for key, value in pruning_mode_results.items()
    }

    post_pruning_mode_combinations = [
        key
        for key, value in pruning_mode_output_predicted.items()
        if value == "similar"
    ]

    logger.info("Starting check similarity in non pruning mode.............")
    for comb in post_pruning_mode_combinations:
        tables = comb.split("_SPLITTER_")
        post_pruning_mode_results[comb] = check_similarity(
            table_infos[tables[0]],
            table_infos[tables[1]],
            pruning_mode=False,
            use_embeddings=False,
        )

    post_pruning_mode_output_predicted = {
        key: ("not_similar" if value[0].score < FINAL_THRESHOLD else "similar")
        for key, value in post_pruning_mode_results.items()
    }
    for i, data_pair in enumerate(post_pruning_mode_results.keys()):
        for key, value in post_pruning_mode_results[data_pair][1].items():
            columns_predicted_scores[key] = value.score

    return (
        pruning_mode_output_predicted,
        post_pruning_mode_output_predicted,
        columns_predicted_scores,
    )


def get_true_predicted_mappings() -> Tuple[
    List[Tuple[str, str, str, str]],
    List[Tuple[str, str, str, str]],
    List[Tuple[str, str, float, str, float, str]],
]:
    table_infos, table_pairs = get_table_infos_and_pairs()
    (
        pruning_mode_output_PREDICTED,
        post_pruning_mode_output_PREDICTED,
        columns_predicted_scores,
    ) = get_similarity_predictions(table_infos=table_infos, table_pairs=table_pairs)

    (
        pruning_table_similarity_labels_expected,
        post_pruning_table_similarity_labels_expected,
        column_similarity_scores_expected,
    ) = load_jsons(INPUT_JSONS_PATH)

    pruning_tables_similarity_mapping_unit_testing_ = (
        get_predicted_expected_similarity_scores_mapping_for_tables(
            pruning_mode_output_PREDICTED, pruning_table_similarity_labels_expected
        )
    )

    post_pruning_tables_similarity_mapping_unit_testing_ = (
        get_predicted_expected_similarity_scores_mapping_for_tables(
            post_pruning_mode_output_PREDICTED,
            post_pruning_table_similarity_labels_expected,
        )
    )

    columns_similarity_mapping_unit_testing_ = (
        get_predicted_expected_similarity_scores_mapping_for_columns(
            columns_predicted_scores,
            column_similarity_scores_expected,
        )
    )
    return (
        pruning_tables_similarity_mapping_unit_testing_,
        post_pruning_tables_similarity_mapping_unit_testing_,
        columns_similarity_mapping_unit_testing_,
    )


(
    pruning_tables_similarity_mapping_unit_testing,
    post_pruning_tables_similarity_mapping_unit_testing,
    columns_similarity_mapping_unit_testing,
) = get_true_predicted_mappings()

############################
# Start unit testing #
############################
# Unit Test for Columns Similarity #
logger.info("--- Unit Test for Columns Similarity ---")


@pytest.mark.parametrize(
    "col_id_1, col_id_2, predicted_score, predicted_label, expected_score, expected_label",
    [
        (a, b, c, d, e, f)
        for a, b, c, d, e, f in columns_similarity_mapping_unit_testing
    ],
)
def test_columns_similarity_public_datasets(
    col_id_1,
    col_id_2,
    predicted_score,
    predicted_label,
    expected_score,
    expected_label,
):
    assert (
        predicted_label == expected_label
    ), f"Test1 failed for column pair: '{(col_id_1, col_id_2)}'"
    if predicted_score is not None and expected_score is not None:
        assert (
            predicted_score >= np.floor(expected_score * 10) / 10
        ), f"Test2 failed for column pair: '{(col_id_1, col_id_2)}'"


# Unit Test for Table Similarity #
@pytest.mark.parametrize(
    "table_id_1, table_id_2, predicted_label, expected_label",
    [(a, b, c, d) for a, b, c, d in pruning_tables_similarity_mapping_unit_testing],
)
def test_pruning_tables_similarity_public_datasets(
    table_id_1,
    table_id_2,
    predicted_label,
    expected_label,
):
    assert (
        predicted_label == expected_label
    ), f"Pruning mode test failed for table pair: '{(table_id_1, table_id_2)}'"


@pytest.mark.parametrize(
    "table_id_1, table_id_2, predicted_label, expected_label",
    [
        (a, b, c, d)
        for a, b, c, d in post_pruning_tables_similarity_mapping_unit_testing
    ],
)
def test_post_pruning_tables_similarity_public_datasets(
    table_id_1,
    table_id_2,
    predicted_label,
    expected_label,
):
    assert (
        predicted_label == expected_label
    ), f"Non Pruning Mode test failed for table pair: '{(table_id_1, table_id_2)}'"
