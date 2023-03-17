import json
import logging
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from datahub_classify.helper_classes import TableInfo
from datahub_classify.similarity_predictor import check_similarity, preprocess_tables

logger = logging.getLogger(__name__)

PRUNING_THRESHOLD = 0.8
FINAL_THRESHOLD = 0.6
COLUMN_SIMILARITY_THRESHOLD = 0.8
CURRENT_WDR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(CURRENT_WDR, "test_input")
TABLE_SIMILARITY_IDEAL_LABELS_PATH = os.path.join(
    INPUT_DIR, "table_similarity_labels_IDEAL.json"
)
COLUMN_SIMILARITY_IDEAL_OUTPUT_PATH = os.path.join(
    INPUT_DIR, "column_similarity_labels_IDEAL.json"
)
TABLE_INFOS_PATH = os.path.join(INPUT_DIR, "table_info_objects.pickle")
TABLE_INFO_COPIES_PATH = os.path.join(INPUT_DIR, "logical_copies.pickle")

PRUNING_TABLE_PAIRS_PATH = os.path.join(INPUT_DIR, "pruning_mode_table_pairs.pkl")
use_embeddings = eval(sys.argv[1]) == 1


def get_table_info_objects() -> Dict[str, TableInfo]:
    with open(TABLE_INFOS_PATH, "rb") as table_info_file:
        table_infos_ = pickle.load(table_info_file)
    with open(TABLE_INFO_COPIES_PATH, "rb") as table_info_copies_file:
        table_info_copies = pickle.load(table_info_copies_file)
    if table_info_copies:
        table_infos_.update(table_info_copies)
    return table_infos_


def get_pruning_mode_table_pairs() -> List[Tuple[str, str]]:
    with open(PRUNING_TABLE_PAIRS_PATH, "rb") as fp:
        table_pairs = pickle.load(fp)
    return table_pairs


def get_table_similarity_ideal_labels():
    with open(TABLE_SIMILARITY_IDEAL_LABELS_PATH) as fp:
        return json.load(fp)


def get_column_similarity_ideal_labels():
    with open(COLUMN_SIMILARITY_IDEAL_OUTPUT_PATH) as fp:
        return json.load(fp)


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
# columns_predicted_scores: Dict[str, float] = dict()
pruning_mode_output_PREDICTED: Dict[str, str] = dict()
non_pruning_mode_output_PREDICTED: Dict[str, str] = dict()
pruning_mode_results: Dict[str, Tuple] = dict()
non_pruning_mode_results: Dict[str, Tuple] = dict()
table_similarity_labels_ideal = get_table_similarity_ideal_labels()
column_similarity_labels_ideal = get_column_similarity_ideal_labels()

logger.info("Creating Table Pairs List................")
pruning_mode_table_pairs = get_pruning_mode_table_pairs()
table_infos = get_table_info_objects()
num_pruning_mode_tables = len(table_infos)

logger.info("Starting check similarity.............")
# pruning_mode_start_time = time.time()
total_pruning_time = 0.0
for table_pair in pruning_mode_table_pairs:
    table_pair_list = sorted(table_pair, key=str.lower)
    table_pair = (table_pair_list[0], table_pair_list[1])
    pruning_mode_start_time = time.time()
    pruning_mode_results[
        f"{table_pair[0]}_SPLITTER_{table_pair[1]}"
    ] = check_similarity(
        table_infos[table_pair[0]],
        table_infos[table_pair[1]],
        pruning_mode=True,
        use_embeddings=False,
    )
    pruning_mode_end_time = time.time()
    total_pruning_time += pruning_mode_end_time - pruning_mode_start_time
# pruning_mode_end_time = time.time()

pruning_mode_output_PREDICTED = {
    key: ("not_similar" if value[0].score < PRUNING_THRESHOLD else "similar")
    for key, value in pruning_mode_results.items()
}

non_pruning_mode_combinations = [
    key for key, value in pruning_mode_output_PREDICTED.items() if value == "similar"
]

non_pruning_table_infos = None
embedding_generation_start_time = 1.0
embedding_generation_end_time = 0.0
non_pruning_table_keys = []
non_pruning_table_infos_list = []
if use_embeddings:
    for pair in non_pruning_mode_combinations:
        tables = pair.split("_SPLITTER_")
        if tables[0] not in non_pruning_table_keys:
            non_pruning_table_keys.append(tables[0])
        if tables[1] not in non_pruning_table_keys:
            non_pruning_table_keys.append(tables[1])

    non_pruning_table_keys = sorted(non_pruning_table_keys)
    non_pruning_table_infos_list = [table_infos[key] for key in non_pruning_table_keys]

    embedding_generation_start_time = time.time()
    non_pruning_table_infos_list = preprocess_tables(non_pruning_table_infos_list)
    embedding_generation_end_time = time.time()

    non_pruning_table_infos = {
        key: value
        for key, value in zip(non_pruning_table_keys, non_pruning_table_infos_list)
    }

if use_embeddings and non_pruning_table_infos:
    table_infos = non_pruning_table_infos

total_non_pruning_time = 0.0
for comb in non_pruning_mode_combinations:
    tables = comb.split("_SPLITTER_")
    non_pruning_mode_start_time = time.time()
    non_pruning_mode_results[comb] = check_similarity(
        table_infos[tables[0]],
        table_infos[tables[1]],
        pruning_mode=False,
        use_embeddings=use_embeddings,
    )
    non_pruning_mode_end_time = time.time()
    total_non_pruning_time += non_pruning_mode_end_time - non_pruning_mode_start_time


embedding_generation_time = (
    embedding_generation_end_time - embedding_generation_start_time
)

non_pruning_mode_output_PREDICTED = {
    key: ("not_similar" if value[0].score < FINAL_THRESHOLD else "similar")
    for key, value in non_pruning_mode_results.items()
}

for i, data_pair in enumerate(non_pruning_mode_results.keys()):
    for column_pair, value in non_pruning_mode_results[data_pair][1].items():
        columns_key = f"{column_pair[0]}_COLSPLITTER_{column_pair[1]}"
        column_similarity_predicted_labels[columns_key] = (
            "not_similar"
            if (value.score is None or value.score < COLUMN_SIMILARITY_THRESHOLD)
            else "similar"
        )
        # columns_predicted_scores[key] = value.score

# ###############################
# # Generate Performance Report #
# ###############################
pruning_report = generate_report_for_table_similarity(
    table_similarity_labels_ideal, pruning_mode_output_PREDICTED
)
final_results = {}
for key in pruning_mode_output_PREDICTED.keys():
    if key in non_pruning_mode_output_PREDICTED:
        final_results[key] = non_pruning_mode_output_PREDICTED[key]
    else:
        final_results[key] = "not_similar"
non_pruning_report = generate_report_for_table_similarity(
    table_similarity_labels_ideal,
    non_pruning_mode_output_PREDICTED,
)
final_table_report = generate_report_for_table_similarity(
    table_similarity_labels_ideal, final_results
)
column_similarity_report = generate_report_for_column_similarity(
    column_similarity_labels_ideal, column_similarity_predicted_labels
)

total_time = (
    total_pruning_time + embedding_generation_time + total_non_pruning_time
    if use_embeddings
    else total_pruning_time + total_non_pruning_time
)
with open("Similarity_predictions_with_embeddings.txt", "a") as file_:
    # TABLE SIMILARITY REPORT:
    file_.write(
        f"-------------------------------------------\n"
        f"PRUNING THRESHOLD: {PRUNING_THRESHOLD}\n"
        f"FINAL THRESHOLD: {FINAL_THRESHOLD}\n"
        f"-------------------------------------------\n"
        f"USE_EMBEDDINGS = {str(use_embeddings)}\n"
        f"Number of Tables {num_pruning_mode_tables}\n"
        f"Total Pruning Time: {total_pruning_time} for {len(pruning_mode_results)} pairs\n"
        f"Total Embedding Time: {embedding_generation_time} for {len(non_pruning_table_keys)} tables\n"
        f"Total Non Pruning Time: {total_non_pruning_time} for {len(non_pruning_mode_results)} pairs\n"
        f"Total Time {total_time}\n\n"
        f"PRUNING MODE CLASSIFICATION REPORT\n{pruning_report[0]}\n{pruning_report[1]}\n"
        f"False Negatives\n"
        f"{pruning_report[2]['fn']}\n"
        f"NON PRUNING MODE CLASSIFICATION REPORT\n"
        f"{non_pruning_report[0]}\n"
        f"{non_pruning_report[1]}\n"
        f"False Negatives\n"
        f"{non_pruning_report[2]['fn']}\n"
        f"FINAL CLASSIFICATION REPORT\n"
        f"{final_table_report[0]}\n"
        f"{final_table_report[1]}\n"
        f"False Negatives\n"
        f"{final_table_report[2]['fn']}\n"
        f"COLUMN SIMILARITY CLASSIFICATION REPORT\n"
        f"{column_similarity_report[0]}\n"
        f"{column_similarity_report[1]}\n"
        f"False Negatives\n"
        f"{column_similarity_report[2]['fn']}\n\n"
    )
