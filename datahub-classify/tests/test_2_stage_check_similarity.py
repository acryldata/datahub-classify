import glob
import itertools
import json
import logging
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix

from datahub_classify.helper_classes import (
    ColumnInfo,
    ColumnMetadata,
    TableInfo,
    TableMetadata,
)
from datahub_classify.similarity_predictor import check_similarity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
PRUNING_THRESHOLD = 0.8
FINAL_THRESHOLD = 0.6
column_similar_threshold = 0.75
column_similarity_of_logical_copies: Dict[str, str] = dict()


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
    predicted_similarity_labels_unit_testing,
    expected_similarity_scores_unit_testing,
):
    """generate mapping of predicted - expected similarity scores, required for unit testing"""

    mapping = []
    for pair in predicted_similarity_scores_unit_testing.keys():
        key_ = f"{pair[0]}_COLSPLITTER_{pair[1]}"
        if key_ in expected_similarity_scores_unit_testing.keys():
            if expected_similarity_scores_unit_testing[key_] is None:
                expected_similarity_label_unit_testing = "None"
            elif (
                expected_similarity_scores_unit_testing[key_]
                >= column_similar_threshold
            ):
                expected_similarity_label_unit_testing = "similar"
            else:
                expected_similarity_label_unit_testing = "not_similar"

            if predicted_similarity_scores_unit_testing[pair] is None:
                predicted_similarity_label_unit_testing = "None"
            elif (
                predicted_similarity_scores_unit_testing[pair]
                >= column_similar_threshold
            ):
                predicted_similarity_label_unit_testing = "similar"
            else:
                predicted_similarity_label_unit_testing = "not_similar"

            mapping.append(
                (
                    pair[0],
                    pair[1],
                    predicted_similarity_scores_unit_testing[pair],
                    predicted_similarity_label_unit_testing,
                    expected_similarity_scores_unit_testing[key_],
                    expected_similarity_label_unit_testing,
                )
            )
    return mapping


logger.info("libraries Imported..................")
SEED = 100
platforms = ["A", "B", "C", "D", "E"]

input_dir = "C:/PROJECTS/Acryl/acryl_git/datahub-classify/tests/datasets"
# output_jsons_dir = "C:/PROJECTS/Acryl/acryl_git/datahub-classify/tests/temp_jsons"
input_jsons_path = "c:/PROJECTS/Acryl/acryl_git/datahub-classify/tests/expected_output"

# table_similarity_labels_pruning_expected: dict(Tuple, str) = {}
# table_similarity_labels_pruning_actual: dict(Tuple, str) = {}
# column_similarity_labels_non_pruning_expected: dict(Tuple, str) = {}
# column_similarity_labels_non_pruning_actual: dict(Tuple, str) = {}

all_datasets_paths = {
    os.path.basename(file).rsplit(".", 1)[0]: file
    for file in glob.glob(f"{input_dir}/*")
}


def generate_report(true_labels, predicted_labels, threshold):
    keys = list(predicted_labels.keys())
    y_pred = [
        0
        if (
            isinstance(predicted_labels[key], int)
            or predicted_labels[key][0].score <= threshold
        )
        else 1
        for key in keys
    ]
    y_true = [0 if true_labels[key] == "not_similar" else 1 for key in keys]
    target_names = ["not_similar", "similar"]
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


def populate_tableinfo_object(dataset_name):
    """populate table info object for a dataset"""
    df = load_df(dataset_name)
    np.random.seed(SEED)
    table_meta_info = {
        "Name": dataset_name,
        "Description": f"This table contains description of {dataset_name}",
        "Platform": platforms[np.random.randint(0, 5)],
        "Table_Id": dataset_name,
    }
    col_infos = []
    for col in df.columns:
        fields = {
            "Name": col,
            "Description": f" {col}",
            "Datatype": str(df[col].dropna().dtype),
            "Dataset_Name": dataset_name,
            "Column_Id": dataset_name + "_SPLITTER_" + col,
        }
        metadata_col = ColumnMetadata(fields)
        # parent_cols = list()
        col_info_ = ColumnInfo(metadata_col)
        col_infos.append(col_info_)

    metadata_table = TableMetadata(table_meta_info)
    # parent_tables = list()
    table_info = TableInfo(metadata_table, col_infos)
    return table_info


def populate_similar_tableinfo_object(dataset_name):
    """populate table info object for a dataset by randomly adding some additional
    columns to the dataset, thus obtain a logical copy of input dataset"""
    df = load_df(dataset_name)
    df.columns = ["data1" + "_" + col for col in df.columns]
    random_df_key = list(all_datasets_paths.keys())[
        np.random.randint(0, len(all_datasets_paths))
    ]
    while random_df_key == dataset_name:
        random_df_key = list(all_datasets_paths.keys())[
            np.random.randint(0, len(all_datasets_paths))
        ]
    random_df = load_df(random_df_key).copy()
    random_df.columns = ["data2" + "_" + col for col in random_df.columns]
    second_df = pd.concat([df, random_df], axis=1)
    cols_to_keep = list(df.columns) + list(random_df.columns[:2])
    second_df = second_df[cols_to_keep]
    np.random.seed(SEED)
    table_meta_info = {
        "Name": dataset_name + "_LOGICAL_COPY",
        "Description": f" {dataset_name}",
        "Platform": platforms[np.random.randint(0, 5)],
        "Table_Id": dataset_name + "_LOGICAL_COPY",
    }
    col_infos = []
    swap_case = ["yes", "no"]
    # fmt:off
    common_variations = ["#", "$", "%", "&", "*", "-", ".", ":", ";", "?",
                         "@", "_", "~", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                         ]
    # fmt:on
    for col in second_df.columns:
        col_name = col.split("_", 1)[1]
        col_name_with_variation = str(
            common_variations[np.random.randint(0, len(common_variations))]
        )
        for word in re.split("[^A-Za-z]", col_name):
            random_variation = str(
                common_variations[np.random.randint(0, len(common_variations))]
            )
            is_swap_case = swap_case[np.random.randint(0, 2)]
            if is_swap_case:
                word = word.swapcase()
            col_name_with_variation = col_name_with_variation + word + random_variation

        fields = {
            "Name": col_name_with_variation,
            "Description": f'{col.split("_", 1)[1]}',
            "Datatype": str(second_df[col].dropna().dtype),
            "Dataset_Name": dataset_name + "_LOGICAL_COPY",
            "Column_Id": dataset_name
            + "_LOGICAL_COPY_"
            + "_SPLITTER_"
            + col.split("_", 1)[1],
        }
        # Update the column similarity labels for logical copies:
        key = f"{dataset_name}_SPLITTER_{col.split('_', 1)[1]}_COLSPLITTER_{fields['Column_Id']}"
        column_similarity_of_logical_copies[key] = "similar"

        metadata_col = ColumnMetadata(fields)
        parent_cols = [col if col in df.columns else None]
        col_info_ = ColumnInfo(metadata_col)
        col_info_.parent_columns = parent_cols
        col_infos.append(col_info_)
    metadata_table = TableMetadata(table_meta_info)
    parent_tables = [dataset_name]
    table_info = TableInfo(metadata_table, col_infos, parent_tables)
    return table_info


model = SentenceTransformer("all-MiniLM-L6-v2")
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
table_infos = {key: populate_tableinfo_object(key) for key in all_datasets_paths.keys()}
table_info_copies = {
    f"{key}_COPY": populate_similar_tableinfo_object(key)
    for key in all_datasets_paths.keys()
}

logger.info("Creating Table Pairs List................")
table_pairs = list(itertools.combinations(table_infos.keys(), 2))
table_infos.update(table_info_copies)
for key in all_datasets_paths.keys():
    table_pairs.append((key, f"{key}_COPY"))

logger.info("Starting check similarity.............")
for pair in table_pairs:
    pruning_mode_results[f"{pair[0]}_SPLITTER_{pair[1]}"] = check_similarity(
        table_infos[pair[0]],
        table_infos[pair[1]],
        pruning_mode=True,
        use_embeddings=False,
    )
pruning_mode_output_PREDICTED = {
    key: ("not_similar" if value[0].score <= PRUNING_THRESHOLD else "similar")
    for key, value in pruning_mode_results.items()
}

# post_pruning_mode_combinations = [key for key in pruning_mode_results.keys() if
#                                   pruning_mode_results[key][0].score > PRUNING_THRESHOLD]
post_pruning_mode_combinations = [
    key for key, value in pruning_mode_output_PREDICTED.items() if value == "similar"
]

for comb in post_pruning_mode_combinations:
    tables = comb.split("_SPLITTER_")
    post_pruning_mode_results[comb] = check_similarity(
        table_infos[tables[0]],
        table_infos[tables[1]],
        pruning_mode=False,
        use_embeddings=False,
    )

post_pruning_mode_output_PREDICTED = {
    key: ("not_similar" if value[0].score <= FINAL_THRESHOLD else "similar")
    for key, value in post_pruning_mode_results.items()
}

pruning_tables_similarity_mapping_unit_testing = (
    get_predicted_expected_similarity_scores_mapping_for_tables(
        pruning_mode_output_PREDICTED, pruning_table_similarity_labels_expected
    )
)
# print(pruning_mode_results["test_SPLITTER_train_COPY"])
columns_correct_preds: List[Tuple] = list()
columns_wrong_preds: List[Tuple] = list()
columns_predicted_scores: Dict[str, float] = dict()
# column_similarity_scores: Dict[Tuple, float] = dict()
columns_predicted_labels: Dict[Tuple, str] = dict()
columns_actual_labels: Dict[Tuple, str] = dict()

for i, data_pair in enumerate(post_pruning_mode_results.keys()):
    for key, value in post_pruning_mode_results[data_pair][1].items():
        columns_predicted_scores[key] = value.score

columns_similarity_mapping_unit_testing = (
    get_predicted_expected_similarity_scores_mapping_for_columns(
        columns_predicted_scores,
        columns_predicted_labels,
        column_similarity_scores_expected,
    )
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
