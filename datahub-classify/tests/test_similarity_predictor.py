import json
import logging
import os

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from datahub_classify.helper_classes import (
    ColumnInfo,
    ColumnMetadata,
    TableInfo,
    TableMetadata,
)
from datahub_classify.similarity_predictor import check_similarity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
current_wdr = os.path.dirname(os.path.abspath(__file__))
input_data_dir = os.path.join(current_wdr, "datasets")
input_jsons_dir = os.path.join(current_wdr, "expected_output")
ideal_json_path = os.path.join(input_jsons_dir, "expected_infotypes_IDEAL.json")

with open(ideal_json_path, "rb") as file:
    ideal_infotypes = json.load(file)
similar_threshold = 0.75
update_columns_expected_similarity_scores_UNIT_TESTING = False
update_tables_expected_similarity_scores_UNIT_TESTING = False
update_tables_expected_similarity_labels_IDEAL = False
SEED = 42

np.random.seed(SEED)


def get_public_data(input_data_path):
    logger.info(f"==============={input_data_path}=================")
    dataset_dict = {}
    for root, dirs, files in os.walk(input_data_path):
        for i, filename in enumerate(files):
            if filename.endswith(".csv"):
                dataset_name = filename.replace(".csv", "")
                dataset_dict[dataset_name] = pd.read_csv(os.path.join(root, filename))
            elif filename.endswith(".xlsx"):
                dataset_name = filename.replace(".xlsx", "")
                dataset_dict[dataset_name] = pd.read_excel(os.path.join(root, filename))
    return dataset_dict


platforms = ["A", "B", "C", "D", "E"]


def populate_tableinfo_object(dataset_name):
    np.random.seed(SEED)
    table_meta_info = {
        "Name": dataset_name,
        "Description": f"This table contains description of {dataset_name}",
        "Platform": platforms[np.random.randint(0, 5)],
        "Table_Id": dataset_name,
    }
    col_infos = []
    for col in public_data_list[dataset_name].columns:
        fields = {
            "Name": col,
            "Description": f" {col}",
            "Datatype": public_data_list[dataset_name][col].dropna().dtype,
            "Dataset_Name": dataset_name,
            "Column_Id": dataset_name + "_SPLITTER_" + col,
        }
        metadata_col = ColumnMetadata(fields)
        # parent_cols = list()
        col_info = ColumnInfo(metadata_col)
        col_infos.append(col_info)

    metadata_table = TableMetadata(table_meta_info)
    # parent_tables = list()
    table_info = TableInfo(metadata_table, col_infos)
    return table_info


def populate_similar_tableinfo_object(dataset_name):
    df = public_data_list[dataset_name].copy()
    random_df_key = list(public_data_list.keys())[
        np.random.randint(0, len(public_data_list))
    ]
    while random_df_key == dataset_name:
        random_df_key = list(public_data_list.keys())[
            np.random.randint(0, len(public_data_list))
        ]
    random_df = public_data_list[random_df_key].copy()
    df.columns = ["data1" + "_" + col for col in df.columns]
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
    for col in second_df.columns:
        fields = {
            "Name": col.split("_", 1)[1],
            "Description": f'{col.split("_", 1)[1]}',
            "Datatype": second_df[col].dropna().dtype,
            "Dataset_Name": dataset_name + "_LOGICAL_COPY",
            "Column_Id": dataset_name
            + "_LOGICAL_COPY_"
            + "_SPLITTER_"
            + col.split("_", 1)[1],
        }
        metadata_col = ColumnMetadata(fields)
        parent_cols = [col if col in df.columns else None]
        col_info = ColumnInfo(metadata_col)
        col_info.parent_columns = parent_cols
        col_infos.append(col_info)
    metadata_table = TableMetadata(table_meta_info)
    parent_tables = [dataset_name]
    table_info = TableInfo(metadata_table, col_infos, parent_tables)
    return table_info


def load_expected_similarity_json(input_jsons_path):
    logger.info("--- Loading JSON files ---")
    with open(
        os.path.join(
            input_jsons_path, "expected_columns_similarity_scores_UNIT_TESTING.json"
        )
    ) as filename:
        expected_columns_similarity_scores_unit_testing = json.load(filename)

    with open(
        os.path.join(
            input_jsons_path, "expected_tables_similarity_scores_UNIT_TESTING.json"
        )
    ) as filename:
        expected_tables_similarity_scores_unit_testing = json.load(filename)

    with open(
        os.path.join(input_jsons_path, "expected_tables_similarity_labels_IDEAL.json")
    ) as filename:
        expected_tables_similarity_labels_ideal = json.load(filename)

    return (
        expected_tables_similarity_scores_unit_testing,
        expected_tables_similarity_labels_ideal,
        expected_columns_similarity_scores_unit_testing,
    )


def get_predicted_expected_similarity_scores_mapping(
    predicted_similarity_scores_unit_testing,
    predicted_similarity_labels_unit_testing,
    expected_similarity_scores_unit_testing,
):
    mapping = []
    for pair in predicted_similarity_scores_unit_testing.keys():
        key_ = pair[0] + "-" + pair[1]
        if expected_similarity_scores_unit_testing.get(key_, None):
            if expected_similarity_scores_unit_testing[key_] >= similar_threshold:
                expected_similarity_label_unit_testing = "similar"
            else:
                expected_similarity_label_unit_testing = "not_similar"
            mapping.append(
                (
                    pair[0],
                    pair[1],
                    predicted_similarity_scores_unit_testing[pair],
                    predicted_similarity_labels_unit_testing[pair],
                    expected_similarity_scores_unit_testing[key_],
                    expected_similarity_label_unit_testing,
                )
            )
    return mapping


def update_scores_and_labels_lists(
    combination,
    overall_table_similarity_score,
    column_similarity_scores,
    columns_predicted_scores,
    columns_predicted_labels,
    columns_correct_preds,
    columns_wrong_preds,
    tables_predicted_scores,
    tables_predicted_labels,
):
    dataset_key_1 = combination[0]
    dataset_key_2 = combination[1]
    tables_predicted_scores[
        (dataset_key_1, dataset_key_2)
    ] = overall_table_similarity_score
    if overall_table_similarity_score > similar_threshold:
        tables_predicted_labels[(dataset_key_1, dataset_key_2)] = "similar"
    else:
        tables_predicted_labels[(dataset_key_1, dataset_key_2)] = "not_similar"
    for col_pair in column_similarity_scores.keys():
        columns_predicted_scores[col_pair] = column_similarity_scores[col_pair]
        col_1 = col_pair[0].split("_SPLITTER_", 1)[1]
        col_2 = col_pair[1].split("_SPLITTER_", 1)[1]
        if ideal_infotypes.get(dataset_key_1, None) and ideal_infotypes.get(
            dataset_key_2, None
        ):
            if ideal_infotypes[dataset_key_1].get(col_1, None) and ideal_infotypes[
                dataset_key_2
            ].get(col_2, None):
                if (
                    ideal_infotypes[dataset_key_1][col_1]
                    == ideal_infotypes[dataset_key_2][col_2]
                ):
                    if column_similarity_scores[col_pair] >= similar_threshold:
                        columns_predicted_labels[col_pair] = "similar"
                        columns_correct_preds.append(
                            (col_pair, column_similarity_scores[col_pair])
                        )
                    else:
                        columns_predicted_labels[col_pair] = "not_similar"
                        columns_wrong_preds.append(
                            (col_pair, column_similarity_scores[col_pair])
                        )
                else:
                    if column_similarity_scores[col_pair] >= similar_threshold:
                        columns_predicted_labels[col_pair] = "similar"
                        columns_wrong_preds.append(
                            (col_pair, column_similarity_scores[col_pair])
                        )
                    else:
                        columns_predicted_labels[col_pair] = "not_similar"
                        columns_correct_preds.append(
                            (col_pair, column_similarity_scores[col_pair])
                        )

    return (
        tables_predicted_scores,
        tables_predicted_labels,
        columns_predicted_scores,
        columns_predicted_labels,
        columns_correct_preds,
        columns_wrong_preds,
    )


logger.info("----------Starting Testing-----------")
public_data_list = get_public_data(input_data_dir)
data_combinations = combinations(public_data_list.keys(), 2)
columns_correct_preds: List[Tuple] = list()
columns_wrong_preds: List[Tuple] = list()
columns_predicted_scores: Dict[Tuple, float] = dict()
columns_predicted_labels: Dict[Tuple, str] = dict()
tables_predicted_scores: Dict[Tuple, float] = dict()
tables_predicted_labels: Dict[Tuple, str] = dict()

# Evaluate Dissimilar Tables #
logger.info("--------Evaluating pairs composed of different tables ------")
for comb in data_combinations:
    dataset_name_1 = comb[0]
    dataset_name_2 = comb[1]
    table_info_1 = populate_tableinfo_object(dataset_name=dataset_name_1)
    table_info_2 = populate_tableinfo_object(dataset_name=dataset_name_2)
    overall_table_similarity_score, column_similarity_scores = check_similarity(
        table_info_1, table_info_2
    )

    (
        tables_predicted_scores,
        tables_predicted_labels,
        columns_predicted_scores,
        columns_predicted_labels,
        columns_correct_preds,
        columns_wrong_preds,
    ) = update_scores_and_labels_lists(
        comb,
        overall_table_similarity_score,
        column_similarity_scores,
        columns_predicted_scores,
        columns_predicted_labels,
        columns_correct_preds,
        columns_wrong_preds,
        tables_predicted_scores,
        tables_predicted_labels,
    )

# Evaluate Similar Tables (Logical Copies) #
logger.info("-----------Evaluating Table pairs that are logical copies--------")
for data in public_data_list.keys():
    dataset_name_1 = data
    table_info_1 = populate_tableinfo_object(dataset_name=dataset_name_1)
    table_info_2 = populate_similar_tableinfo_object(dataset_name=dataset_name_1)
    overall_table_similarity_score, column_similarity_scores = check_similarity(
        table_info_1, table_info_2
    )
    comb = (dataset_name_1, dataset_name_1 + "_LOGICAL_COPY")
    (
        tables_predicted_scores,
        tables_predicted_labels,
        columns_predicted_scores,
        columns_predicted_labels,
        columns_correct_preds,
        columns_wrong_preds,
    ) = update_scores_and_labels_lists(
        comb,
        overall_table_similarity_score,
        column_similarity_scores,
        columns_predicted_scores,
        columns_predicted_labels,
        columns_correct_preds,
        columns_wrong_preds,
        tables_predicted_scores,
        tables_predicted_labels,
    )

logger.info("-------Test Statistics-------------")
logger.info(f"Correct predictions (Column Similarity) : {len(columns_correct_preds)}")
logger.info(f"Wrong predictions (Column Similarity) : {len(columns_wrong_preds)}")
logger.info(
    f"Accuracy: {np.round(len(columns_correct_preds) / (len(columns_wrong_preds) + len(columns_correct_preds)), 2)}"
)

if update_columns_expected_similarity_scores_UNIT_TESTING:
    expected_columns_similarity_scores = {}
    for key in columns_predicted_scores.keys():
        col_1 = key[0].split("_SPLITTER_", 1)[1]
        dataset_key_1 = key[0].split("_SPLITTER_", 1)[0]
        col_2 = key[1].split("_SPLITTER_", 1)[1]
        dataset_key_2 = key[1].split("_SPLITTER_", 1)[0]
        if ideal_infotypes.get(dataset_key_1, None) and ideal_infotypes.get(
            dataset_key_2, None
        ):
            if ideal_infotypes[dataset_key_1].get(col_1, None) and ideal_infotypes[
                dataset_key_2
            ].get(col_2, None):
                expected_columns_similarity_scores[key[0] + "-" + key[1]] = float(
                    columns_predicted_scores[key]
                )
    logger.info("Updating expected_columns_similarity_scores_UNIT_TESTING json..")
    with open(
        os.path.join(
            input_jsons_dir, "expected_columns_similarity_scores_UNIT_TESTING.json"
        ),
        "w",
    ) as filename:
        json.dump(expected_columns_similarity_scores, filename, indent=4)

if update_tables_expected_similarity_scores_UNIT_TESTING:
    expected_tables_similarity_scores = {}
    for key in tables_predicted_scores.keys():
        expected_tables_similarity_scores[key[0] + "-" + key[1]] = float(
            tables_predicted_scores[key]
        )
    logger.info("Updating expected_tables_similarity_scores_UNIT_TESTING json..")
    with open(
        os.path.join(
            input_jsons_dir, "expected_tables_similarity_scores_UNIT_TESTING.json"
        ),
        "w",
    ) as filename:
        json.dump(expected_tables_similarity_scores, filename, indent=4)


if update_tables_expected_similarity_labels_IDEAL:
    expected_tables_similarity_labels = {}
    for key in tables_predicted_scores.keys():
        if tables_predicted_scores[key] >= similar_threshold:
            expected_tables_similarity_labels[key[0] + "-" + key[1]] = "similar"
        else:
            expected_tables_similarity_labels[key[0] + "-" + key[1]] = "not_similar"
    logger.info("Updating expected_tables_similarity_labels_IDEAL json..")
    with open(
        os.path.join(input_jsons_dir, "expected_tables_similarity_labels_IDEAL.json"),
        "w",
    ) as filename:
        json.dump(expected_tables_similarity_labels, filename, indent=4)


(
    expected_tables_similarity_scores_unit_testing,
    expected_tables_similarity_labels_ideal,
    expected_columns_similarity_scores_unit_testing,
) = load_expected_similarity_json(input_jsons_dir)

columns_similarity_mapping_unit_testing = (
    get_predicted_expected_similarity_scores_mapping(
        columns_predicted_scores,
        columns_predicted_labels,
        expected_columns_similarity_scores_unit_testing,
    )
)

tables_similarity_mapping_unit_testing = (
    get_predicted_expected_similarity_scores_mapping(
        tables_predicted_scores,
        tables_predicted_labels,
        expected_tables_similarity_scores_unit_testing,
    )
)

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
    assert (
        predicted_score >= np.floor(expected_score * 10) / 10
    ), f"Test2 failed for column pair: '{(col_id_1, col_id_2)}'"


# Unit Test for Tables Similarity #
logger.info("-- Unit Test for Tables Similarity --")


@pytest.mark.parametrize(
    "table_id_1, table_id_2, predicted_score, predicted_label, expected_score, expected_label",
    [(a, b, c, d, e, f) for a, b, c, d, e, f in tables_similarity_mapping_unit_testing],
)
def test_tables_similarity_public_datasets(
    table_id_1,
    table_id_2,
    predicted_score,
    predicted_label,
    expected_score,
    expected_label,
):
    assert (
        predicted_label == expected_label
    ), f"Test1 failed for column pair: '{(table_id_1, table_id_2)}'"
    assert (
        predicted_score >= np.floor(expected_score * 10) / 10
    ), f"Test2 failed for column pair: '{(table_id_1, table_id_2)}'"
