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
INPUT_DIR = os.path.join(CURRENT_WDR, "test_input")
PRUNING_TABLE_SIMILARITY_EXPECTED_LABELS_PATH = os.path.join(
    INPUT_DIR, "pruning_table_similarity_labels_EXPECTED.json"
)
NON_PRUNING_TABLE_SIMILARITY_EXPECTED_LABELS_PATH = os.path.join(
    INPUT_DIR, "non_pruning_table_similarity_labels_EXPECTED.json"
)
COLUMN_SIMILARITY_EXPECTED_OUTPUT_PATH = os.path.join(
    INPUT_DIR, "column_similarity_scores_EXPECTED.json"
)
TABLE_INFOS_PATH = os.path.join(INPUT_DIR, "table_info_objects.pickle")
TABLE_INFO_COPIES_PATH = os.path.join(INPUT_DIR, "logical_copies.pickle")

PRUNING_TABLE_PAIRS_PATH = os.path.join(INPUT_DIR, "pruning_mode_table_pairs.pkl")
NON_PRUNING_TABLE_PAIRS_PATH = os.path.join(
    INPUT_DIR, "non_pruning_mode_table_pairs.pkl"
)


def get_table_info_objects() -> Dict[str, TableInfo]:
    with open(TABLE_INFOS_PATH, "rb") as table_info_file:
        table_infos = pickle.load(table_info_file)
    with open(TABLE_INFO_COPIES_PATH, "rb") as table_info_copies_file:
        table_info_copies = pickle.load(table_info_copies_file)
    if table_info_copies:
        table_infos.update(table_info_copies)
    return table_infos


def get_pruning_mode_table_pairs() -> List[Tuple[str, str]]:
    with open(PRUNING_TABLE_PAIRS_PATH, "rb") as fp:
        table_pairs = pickle.load(fp)
    return table_pairs


def get_non_pruning_mode_table_pairs() -> List[str]:
    with open(NON_PRUNING_TABLE_PAIRS_PATH, "rb") as fp:
        table_pairs = pickle.load(fp)
    return table_pairs


def get_table_pair_expected_similarity_label(key: str, pruning: bool) -> str:
    """generate mapping of predicted - expected similarity scores, required for unit testing"""
    if pruning:
        expected_labels_json_path = PRUNING_TABLE_SIMILARITY_EXPECTED_LABELS_PATH
    else:
        expected_labels_json_path = NON_PRUNING_TABLE_SIMILARITY_EXPECTED_LABELS_PATH
    with open(expected_labels_json_path) as file_:
        table_similarity_labels_expected = json.load(file_)

    expected_similarity_label = table_similarity_labels_expected[key]
    return expected_similarity_label


def get_column_pair_expected_similarity_label(
    column_pair: Tuple[str, str], predicted_score: float
) -> Tuple[str, str, float, float]:
    with open(COLUMN_SIMILARITY_EXPECTED_OUTPUT_PATH) as filename_:
        column_similarity_scores_expected = json.load(filename_)

    key = f"{column_pair[0]}_COLSPLITTER_{column_pair[1]}"
    if column_similarity_scores_expected[key] is None:
        expected_label = "not_similar"
        expected_score = 0.0
    elif column_similarity_scores_expected[key] >= COLUMN_SIMILARITY_THRESHOLD:
        expected_label = "similar"
        expected_score = column_similarity_scores_expected[key]
    else:
        expected_label = "not_similar"
        expected_score = column_similarity_scores_expected[key]
    if predicted_score is None:
        predicted_label = "not_similar"
        predicted_score = 0.0
    elif predicted_score >= COLUMN_SIMILARITY_THRESHOLD:
        predicted_label = "similar"
    else:
        predicted_label = "not_similar"
    return (
        predicted_label,
        expected_label,
        predicted_score,
        expected_score,
    )


############################
# Start unit testing #
############################
# Unit Test for Table Similarity #
@pytest.mark.parametrize(
    "table_1, table_2",
    [(a, b) for a, b in get_pruning_mode_table_pairs()],
)
def test_pruning_tables_similarity_public_datasets(
    table_1,
    table_2,
):
    table_pair = f"{table_1}_SPLITTER_{table_2}"
    table_infos = get_table_info_objects()
    table_id_1 = table_infos[table_1].metadata.table_id
    table_id_2 = table_infos[table_2].metadata.table_id

    table_1_info = table_infos[table_1]
    table_2_info = table_infos[table_2]
    table_similarity_score, _ = check_similarity(
        table_1_info, table_2_info, pruning_mode=True, use_embeddings=False
    )

    predicted_label = (
        "not_similar" if table_similarity_score.score < PRUNING_THRESHOLD else "similar"
    )
    expected_label = get_table_pair_expected_similarity_label(table_pair, pruning=True)
    assert (
        predicted_label == expected_label
    ), f"Pruning mode test failed for table pair: '{(table_id_1, table_id_2)}'"


@pytest.mark.parametrize(
    "table_1, table_2",
    [tuple(pair.split("_SPLITTER_", 1)) for pair in get_non_pruning_mode_table_pairs()],
)
def test_non_pruning_tables_similarity_public_datasets(
    table_1,
    table_2,
):
    table_pair = f"{table_1}_SPLITTER_{table_2}"
    table_infos = get_table_info_objects()
    table_id_1 = table_infos[table_1].metadata.table_id
    table_id_2 = table_infos[table_2].metadata.table_id

    table_1_info = table_infos[table_1]
    table_2_info = table_infos[table_2]
    table_similarity_score, column_similarity_scores = check_similarity(
        table_1_info, table_2_info, pruning_mode=False, use_embeddings=False
    )

    table_predicted_label = (
        "not_similar" if table_similarity_score.score < FINAL_THRESHOLD else "similar"
    )
    table_expected_label = get_table_pair_expected_similarity_label(
        table_pair, pruning=False
    )

    assert (
        table_predicted_label == table_expected_label
    ), f"Non Pruning Mode test failed for table pair: '{(table_id_1, table_id_2)}'"

    for column_pair, value in column_similarity_scores.items():
        col_id_1, col_id_2 = column_pair
        column_predicted_score = value.score
        (
            column_predicted_label,
            column_expected_label,
            column_predicted_score,
            column_expected_score,
        ) = get_column_pair_expected_similarity_label(
            column_pair, column_predicted_score
        )

        assert (
            column_predicted_label == column_expected_label
        ), f"Test1 failed for column pair: '{(col_id_1, col_id_2)}'"
        if column_predicted_score is not None and column_expected_score is not None:
            assert (
                column_predicted_score >= np.floor(column_expected_score * 10) / 10
            ), f"Test2 failed for column pair: '{(col_id_1, col_id_2)}'"
