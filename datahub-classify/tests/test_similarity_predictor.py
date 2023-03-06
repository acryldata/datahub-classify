import glob
import itertools
import json
import logging
import os
import pickle
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import classification_report, confusion_matrix

from datahub_classify.similarity_predictor import check_similarity

logger = logging.getLogger(__name__)

PRUNING_THRESHOLD = 0.8
FINAL_THRESHOLD = 0.6
COLUMN_SIMILARITY_THRESHOLD = 0.8
PLATFORMS = ["A", "B", "C", "D", "E"]


def load_df(dataset_name):
    path = all_datasets_paths[dataset_name]
    if path.endswith("csv"):
        df = pd.read_csv(path, nrows=2)
    elif path.endswith("xlsx"):
        df = pd.read_excel(path, nrows=2)
    else:
        df = None
    return df


def load_jsons(input_jsons_dir):
    with open(
        os.path.join(input_jsons_dir, "table_similarity_labels_IDEAL.json")
    ) as filename:
        table_similarity_labels_ideal_ = json.load(filename)
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
        os.path.join(input_jsons_path, "column_similarity_scores_EXPECTED.json")
    ) as filename_:
        column_similarity_scores_expected_ = json.load(filename_)

    with open(
        os.path.join(input_jsons_path, "column_similarity_labels_IDEAL.json")
    ) as filename_:
        column_similarity_labels_ideal_ = json.load(filename_)

    return (
        table_similarity_labels_ideal_,
        pruning_table_similarity_labels_expected_,
        post_pruning_table_similarity_labels_expected_,
        column_similarity_scores_expected_,
        column_similarity_labels_ideal_,
    )


def get_predicted_expected_similarity_scores_mapping_for_tables(
    predicted_similarity_labels_unit_testing,
    expected_similarity_labels_unit_testing,
):
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
    predicted_similarity_scores_unit_testing,
    expected_similarity_scores_unit_testing,
):
    """generate mapping of predicted - expected similarity scores, required for unit testing"""

    mapping = []
    for pair in predicted_similarity_scores_unit_testing.keys():
        key_ = f"{pair[0]}_COLSPLITTER_{pair[1]}"
        if key_ in expected_similarity_scores_unit_testing.keys():
            if expected_similarity_scores_unit_testing[key_] is None:
                expected_similarity_label = "not_similar"
                expected_similarity_score = 0
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
                predicted_similarity_score = 0
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
            column_similarity_predicted_labels[key_] = predicted_similarity_label
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


def generate_report_for_table_similarity(true_labels, predicted_labels):
    keys = list(predicted_labels.keys())
    y_pred = [0 if predicted_labels[key] == "not_similar" else 1 for key in keys]
    y_true = [0 if true_labels[key] == "not_similar" else 1 for key in keys]
    target_names = [
        "not_similar" if label == 0 else "similar"
        for label in np.unique(y_pred + y_true)
    ]
    misclassification_report: Dict[str, list] = {"tp": [], "fp": [], "tn": [], "fn": []}
    for i in range(len(keys)):
        if y_true[i] == 0 and y_pred[i] == 0:
            misclassification_report["tn"].append(keys[i])
        elif y_true[i] == 0 and y_pred[i] == 1:
            misclassification_report["fp"].append(keys[i])
        elif y_true[i] == 1 and y_pred[i] == 0:
            misclassification_report["fn"].append(keys[i])
        else:
            misclassification_report["tp"].append(keys[i])
    return (
        confusion_matrix(y_true, y_pred),
        classification_report(y_true, y_pred, target_names=target_names),
        misclassification_report,
    )


def generate_report_for_column_similarity(true_labels, predicted_labels):
    keys = list(predicted_labels.keys())
    y_pred_labels = []
    y_true_labels = []
    for key in keys:
        y_pred_labels.append(predicted_labels[key])
        if key not in true_labels.keys():
            y_true_labels.append("not_similar")
        else:
            y_true_labels.append(true_labels[key])

    y_pred = [0 if label == "not_similar" else 1 for label in y_pred_labels]
    y_true = [0 if label == "not_similar" else 1 for label in y_true_labels]
    target_names = [
        "not_similar" if label == 0 else "similar"
        for label in np.unique(y_pred + y_true)
    ]
    misclassification_report: Dict[str, list] = {"tp": [], "fp": [], "tn": [], "fn": []}
    for i in range(len(keys)):
        if y_true[i] == 0 and y_pred[i] == 0:
            misclassification_report["tn"].append(keys[i])
        elif y_true[i] == 0 and y_pred[i] == 1:
            misclassification_report["fp"].append(keys[i])
        elif y_true[i] == 1 and y_pred[i] == 0:
            misclassification_report["fn"].append(keys[i])
        else:
            misclassification_report["tp"].append(keys[i])
    return (
        confusion_matrix(y_true, y_pred),
        classification_report(y_true, y_pred, target_names=target_names),
        misclassification_report,
    )


column_similarity_predicted_labels: Dict[str, str] = dict()
columns_predicted_scores: Dict[str, float] = dict()

current_wdr = os.path.dirname(os.path.abspath(__file__))

input_dir = os.path.join(current_wdr, "datasets")
input_jsons_path = os.path.join(current_wdr, "expected_output")
table_info_copies_path = os.path.join(current_wdr, "logical_copies.pickle")

all_datasets_paths = {
    os.path.basename(file).rsplit(".", 1)[0]: file
    for file in glob.glob(f"{input_dir}/*")
}

pruning_mode_output_PREDICTED: Dict[str, str] = dict()
post_pruning_mode_output_PREDICTED: Dict[str, str] = dict()
pruning_mode_results: Dict[str, Tuple] = dict()
post_pruning_mode_results: Dict[str, Tuple] = dict()

(
    table_similarity_labels_ideal,
    pruning_table_similarity_labels_expected,
    post_pruning_table_similarity_labels_expected,
    column_similarity_scores_expected,
    column_similarity_labels_ideal,
) = load_jsons(input_jsons_path)

logger.info("Creating Tables Info Objects.............")

with open(
    os.path.join(current_wdr, "table_info_objects.pickle"), "rb"
) as table_info_file:
    table_infos = pickle.load(table_info_file)
if os.path.exists(table_info_copies_path):
    with open(
        os.path.join(current_wdr, "logical_copies.pickle"), "rb"
    ) as table_info_copies_file:
        table_info_copies = pickle.load(table_info_copies_file)
else:
    table_info_copies = None

logger.info("Creating Table Pairs List................")
table_pairs = list(itertools.combinations(table_infos.keys(), 2))
if table_info_copies:
    table_info_copies = {
        key: value
        for key, value in table_info_copies.items()
        if key[:-13] in all_datasets_paths.keys()
    }
    table_infos.update(table_info_copies)
for key in all_datasets_paths.keys():
    table_pairs.append((key, f"{key}_LOGICAL_COPY"))

logger.info("Starting check similarity in pruning mode.............")
pruning_mode_start_time = time.time()
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
pruning_mode_end_time = time.time()

pruning_mode_output_PREDICTED = {
    key: ("not_similar" if value[0].score < PRUNING_THRESHOLD else "similar")
    for key, value in pruning_mode_results.items()
}

post_pruning_mode_combinations = [
    key for key, value in pruning_mode_output_PREDICTED.items() if value == "similar"
]

logger.info("Starting check similarity in non pruning mode.............")
post_pruning_mode_start_time = time.time()
for comb in post_pruning_mode_combinations:
    tables = comb.split("_SPLITTER_")
    post_pruning_mode_results[comb] = check_similarity(
        table_infos[tables[0]],
        table_infos[tables[1]],
        pruning_mode=False,
        use_embeddings=False,
    )
post_pruning_mode_end_time = time.time()

post_pruning_mode_output_PREDICTED = {
    key: ("not_similar" if value[0].score < FINAL_THRESHOLD else "similar")
    for key, value in post_pruning_mode_results.items()
}

pruning_tables_similarity_mapping_unit_testing = (
    get_predicted_expected_similarity_scores_mapping_for_tables(
        pruning_mode_output_PREDICTED, pruning_table_similarity_labels_expected
    )
)

for i, data_pair in enumerate(post_pruning_mode_results.keys()):
    for key, value in post_pruning_mode_results[data_pair][1].items():
        columns_predicted_scores[key] = value.score

columns_similarity_mapping_unit_testing = (
    get_predicted_expected_similarity_scores_mapping_for_columns(
        columns_predicted_scores,
        column_similarity_scores_expected,
    )
)

total_pruning_time = pruning_mode_end_time - pruning_mode_start_time
total_post_pruning_time = post_pruning_mode_end_time - post_pruning_mode_start_time
total_check_similarity_time = total_post_pruning_time + total_pruning_time

#
# ###############################
# # Generate Performance Report #
# ###############################
pruning_report = generate_report_for_table_similarity(
    table_similarity_labels_ideal,
    pruning_mode_output_PREDICTED,
)
final_results = {}
for key in pruning_mode_output_PREDICTED.keys():
    if key in post_pruning_mode_output_PREDICTED:
        final_results[key] = post_pruning_mode_output_PREDICTED[key]
    else:
        final_results[key] = "not_similar"
post_pruning_report = generate_report_for_table_similarity(
    table_similarity_labels_ideal, post_pruning_mode_output_PREDICTED
)
final_table_report = generate_report_for_table_similarity(
    table_similarity_labels_ideal, final_results
)
column_similarity_report = generate_report_for_column_similarity(
    column_similarity_labels_ideal, column_similarity_predicted_labels
)

with open("Similarity_predictions.txt", "w") as file_:
    """TABLE AND COLUMN SIMILARITY REPORT:"""
    file_.write(
        f"-------------------------------------------\n"
        f"PRUNING THRESHOLD: {PRUNING_THRESHOLD}\n"
        f"FINAL THRESHOLD: {FINAL_THRESHOLD}\n"
        f"-------------------------------------------\n"
        f"Number of Tables {len(table_infos)}\n"
        f"Total Pruning Time: {total_pruning_time} for {len(pruning_mode_results)} pairs\n"
        f"Total Post Pruning Time: {total_post_pruning_time} for {len(post_pruning_mode_results)} pairs\n"
        f"Total Time {total_check_similarity_time}\n"
        f"PRUNING MODE CLASSIFICATION REPORT\n{pruning_report[0]}\n{pruning_report[1]}\n"
        f"False Negatives\n"
        f"{pruning_report[2]['fn']}\n"
        f"POST PRUNING MODE CLASSIFICATION REPORT\n"
        f"{post_pruning_report[0]}\n"
        f"{post_pruning_report[1]}\n"
        f"False Negatives\n"
        f"{post_pruning_report[2]['fn']}\n"
        f"FINAL CLASSIFICATION REPORT\n"
        f"{final_table_report[0]}\n"
        f"{final_table_report[1]}\n"
        f"False Negatives\n"
        f"{final_table_report[2]['fn']}\n"
        f"COLUMN SIMILARITY CLASSIFICATION REPORT\n"
        f"{column_similarity_report[0]}\n"
        f"{column_similarity_report[1]}\n"
        f"False Negatives\n"
        f"{column_similarity_report[2]['fn']}"
    )

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


post_pruning_tables_similarity_mapping_unit_testing = (
    get_predicted_expected_similarity_scores_mapping_for_tables(
        post_pruning_mode_output_PREDICTED,
        post_pruning_table_similarity_labels_expected,
    )
)


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
