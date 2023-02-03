import json
import logging
import os
import pickle
import re

# import pickle
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from datahub_classify.helper_classes import (
    ColumnInfo,
    ColumnMetadata,
    TableInfo,
    TableMetadata,
)
from datahub_classify.similarity_predictor import check_similarity, preprocess_tables

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
current_wdr = os.path.dirname(os.path.abspath(__file__))
input_data_dir = os.path.join(current_wdr, "datasets")
input_jsons_dir = os.path.join(current_wdr, "expected_output")

similar_threshold = 0.75
update_columns_expected_similarity_scores_UNIT_TESTING = False
update_tables_expected_similarity_scores_UNIT_TESTING = False
update_tables_expected_similarity_labels_IDEAL = False
SEED = 42
np.random.seed(SEED)

platforms = ["A", "B", "C", "D", "E"]


def get_public_data(input_data_path):
    """load public datasets from directory"""

    logger.info(f"==============={input_data_path}=================")
    dataset_dict = {}
    for root, dirs, files in os.walk(input_data_path):
        for i, filename_ in enumerate(files):
            if filename_.endswith(".csv"):
                dataset_name = filename_.replace(".csv", "")
                dataset_dict[dataset_name] = pd.read_csv(os.path.join(root, filename_))
            elif filename_.endswith(".xlsx"):
                dataset_name = filename_.replace(".xlsx", "")
                dataset_dict[dataset_name] = pd.read_excel(
                    os.path.join(root, filename_)
                )
            # if i > 3:
            #     break
    return dataset_dict


def populate_tableinfo_object(dataset_name):
    """populate table info object for a dataset"""

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
            "Datatype": str(public_data_list[dataset_name][col].dropna().dtype),
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
    swap_case = ["yes", "no"]
    common_variations = [
        "#",
        "$",
        "%",
        "&",
        "*",
        "-",
        ".",
        ":",
        ";",
        "?",
        "@",
        "_",
        "~",
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]
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
        metadata_col = ColumnMetadata(fields)
        parent_cols = [col if col in df.columns else None]
        col_info_ = ColumnInfo(metadata_col)
        col_info_.parent_columns = parent_cols
        col_infos.append(col_info_)
    metadata_table = TableMetadata(table_meta_info)
    parent_tables = [dataset_name]
    table_info = TableInfo(metadata_table, col_infos, parent_tables)
    return table_info


def load_json_files(input_jsons_path):
    """load ideal and unit testing JSON files. For columns, infotypes_IDEAL
    serves as the ideal JSON"""

    logger.info("--- Loading JSON files ---")
    with open(
        os.path.join(
            input_jsons_path, "expected_columns_similarity_scores_UNIT_TESTING.json"
        )
    ) as filename_:
        expected_columns_similarity_scores_unit_testing_ = json.load(filename_)

    with open(
        os.path.join(
            input_jsons_path, "expected_tables_similarity_scores_UNIT_TESTING.json"
        )
    ) as filename_:
        expected_tables_similarity_scores_unit_testing_ = json.load(filename_)

    with open(
        os.path.join(input_jsons_path, "expected_tables_similarity_labels_IDEAL.json")
    ) as filename_:
        expected_tables_similarity_labels_ideal_ = json.load(filename_)

    with open(
        os.path.join(input_jsons_path, "expected_infotypes_IDEAL.json")
    ) as filename_:
        ideal_infotypes_ = json.load(filename_)

    return (
        expected_tables_similarity_scores_unit_testing_,
        expected_tables_similarity_labels_ideal_,
        expected_columns_similarity_scores_unit_testing_,
        ideal_infotypes_,
    )


def get_predicted_expected_similarity_scores_mapping(
    predicted_similarity_scores_unit_testing,
    predicted_similarity_labels_unit_testing,
    expected_similarity_scores_unit_testing,
):
    """generate mapping of predicted - expected similarity scores, required for unit testing"""

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


def update_scores_and_labels_dicts(
    combination,
    overall_table_similarity_score_,
    column_similarity_scores_,
    columns_predicted_scores_,
    columns_predicted_labels_,
    columns_actual_labels_,
    columns_correct_preds_,
    columns_wrong_preds_,
    tables_predicted_scores_,
    tables_predicted_labels_,
    tables_actual_labels_,
):
    """update dictionaries with predicted scores, labels and actual labels"""

    dataset_key_1_ = combination[0]
    dataset_key_2_ = combination[1]
    tables_predicted_scores_[
        (dataset_key_1_, dataset_key_2_)
    ] = overall_table_similarity_score_
    tables_actual_labels_[
        (dataset_key_1_, dataset_key_2_)
    ] = expected_tables_similarity_labels_ideal[dataset_key_1_ + "-" + dataset_key_2_]
    if overall_table_similarity_score_ > similar_threshold:
        tables_predicted_labels_[(dataset_key_1_, dataset_key_2_)] = "similar"
    else:
        tables_predicted_labels_[(dataset_key_1_, dataset_key_2_)] = "not_similar"
    for col_pair_ in column_similarity_scores_.keys():
        columns_predicted_scores_[col_pair_] = column_similarity_scores_[col_pair_]
        col_1_ = col_pair_[0].split("_SPLITTER_", 1)[1]
        col_2_ = col_pair_[1].split("_SPLITTER_", 1)[1]
        if ideal_infotypes.get(dataset_key_1_, None) and ideal_infotypes.get(
            dataset_key_2_, None
        ):
            if ideal_infotypes[dataset_key_1_].get(col_1_, None) and ideal_infotypes[
                dataset_key_2_
            ].get(col_2_, None):
                if (
                    ideal_infotypes[dataset_key_1_][col_1_]
                    == ideal_infotypes[dataset_key_2_][col_2_]
                ):
                    columns_actual_labels_[col_pair_] = "similar"
                    if column_similarity_scores_[col_pair_] >= similar_threshold:
                        columns_predicted_labels_[col_pair_] = "similar"
                        columns_correct_preds_.append(
                            (col_pair_, column_similarity_scores_[col_pair_])
                        )
                    else:
                        columns_predicted_labels_[col_pair_] = "not_similar"
                        columns_wrong_preds_.append(
                            (col_pair_, column_similarity_scores_[col_pair_])
                        )
                else:
                    columns_actual_labels_[col_pair_] = "not_similar"
                    if column_similarity_scores_[col_pair_] >= similar_threshold:
                        columns_predicted_labels_[col_pair_] = "similar"
                        columns_wrong_preds_.append(
                            (col_pair_, column_similarity_scores_[col_pair_])
                        )
                    else:
                        columns_predicted_labels_[col_pair_] = "not_similar"
                        columns_correct_preds_.append(
                            (col_pair_, column_similarity_scores_[col_pair_])
                        )

    return (
        tables_predicted_scores_,
        tables_predicted_labels_,
        tables_actual_labels_,
        columns_predicted_scores_,
        columns_predicted_labels_,
        columns_actual_labels_,
        columns_correct_preds_,
        columns_wrong_preds_,
    )


def get_prediction_statistics(predicted_labels, actual_labels, entity):
    """generate precision, recall and confusion matrix for columns and tables similarity predictions"""

    y_true = list(actual_labels.values())
    y_pred = list(predicted_labels.values())
    prediction_stats = pd.DataFrame()
    prediction_stats["label"] = ["similar", "not_similar"]
    prediction_stats["Precision"] = np.round(
        precision_score(
            y_true, y_pred, average=None, labels=["similar", "not_similar"]
        ),
        2,
    )
    prediction_stats["Recall"] = np.round(
        recall_score(y_true, y_pred, average=None, labels=["similar", "not_similar"]), 2
    )
    df_confusion_matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=["similar", "not_similar"]),
        columns=["similar_predicted", "not_similar_predicted"],
        index=["similar_actual", "not_similar_actual"],
    )
    logger.info(
        f"*************Prediction Statistics for {entity} ***************************"
    )
    logger.info(prediction_stats)
    logger.info("********************")
    logger.info(df_confusion_matrix)
    prediction_stats.to_csv(
        f"Prediction_statistics_{entity}_{similar_threshold}.csv", index=False
    )
    df_confusion_matrix.to_csv(
        f"confusion_matrix_{entity}_{similar_threshold}.csv", index=False
    )


# Start testing
logger.info("----------Starting Testing-----------")
public_data_list = get_public_data(input_data_dir)
data_combinations = combinations(public_data_list.keys(), 2)

(
    expected_tables_similarity_scores_unit_testing,
    expected_tables_similarity_labels_ideal,
    expected_columns_similarity_scores_unit_testing,
    ideal_infotypes,
) = load_json_files(input_jsons_dir)

columns_correct_preds: List[Tuple] = list()
columns_wrong_preds: List[Tuple] = list()
columns_predicted_scores: Dict[Tuple, float] = dict()
columns_predicted_labels: Dict[Tuple, str] = dict()
columns_actual_labels: Dict[Tuple, str] = dict()
tables_predicted_scores: Dict[Tuple, float] = dict()
tables_predicted_labels: Dict[Tuple, str] = dict()
tables_actual_labels: Dict[Tuple, str] = dict()

# Evaluate Dissimilar Tables #
logger.info("--------Evaluating pairs composed of different tables ------")
for comb in data_combinations:
    dataset_name_1 = comb[0]
    dataset_name_2 = comb[1]
    table_info_1 = populate_tableinfo_object(dataset_name=dataset_name_1)
    table_info_2 = populate_tableinfo_object(dataset_name=dataset_name_2)
    table_info_list = preprocess_tables([table_info_1, table_info_2])
    table_similarity_info, column_similarity_info = check_similarity(
        table_info_list[0], table_info_list[1]
    )
    overall_table_similarity_score = table_similarity_info.score
    column_similarity_scores = {}
    for key, value in column_similarity_info.items():
        column_similarity_scores[key] = value.score

    (
        tables_predicted_scores,
        tables_predicted_labels,
        tables_actual_labels,
        columns_predicted_scores,
        columns_predicted_labels,
        columns_actual_labels,
        columns_correct_preds,
        columns_wrong_preds,
    ) = update_scores_and_labels_dicts(
        comb,
        overall_table_similarity_score,
        column_similarity_scores,
        columns_predicted_scores,
        columns_predicted_labels,
        columns_actual_labels,
        columns_correct_preds,
        columns_wrong_preds,
        tables_predicted_scores,
        tables_predicted_labels,
        tables_actual_labels,
    )

# Evaluate Similar Tables (Logical Copies) #
logger.info("-----------Evaluating Table pairs that are logical copies--------")
column_similarity_with_variation_ideal_labels = {}
column_similarity_with_variation_predicted_labels = {}
wrong_preds = {}

for data in public_data_list.keys():
    dataset_name_1 = data
    table_info_1 = populate_tableinfo_object(dataset_name=dataset_name_1)
    table_info_2 = populate_similar_tableinfo_object(dataset_name=dataset_name_1)
    column_truename_changedname_mapping = {}
    for col_info in table_info_2.column_infos:
        true_name = col_info.metadata.column_id.split("_SPLITTER_", 1)[1]
        changed_name = col_info.metadata.name
        column_truename_changedname_mapping[true_name] = changed_name
    table_info_list = preprocess_tables([table_info_1, table_info_2])
    table_similarity_info, column_similarity_info = check_similarity(
        table_info_list[0], table_info_list[1]
    )
    overall_table_similarity_score = table_similarity_info.score
    column_similarity_scores = {}
    for key, value in column_similarity_info.items():
        column_similarity_scores[key] = value.score

    for col_pair in column_similarity_scores.keys():
        col_1_true_name = col_pair[0].split("_SPLITTER_", 1)[1]
        col_2_true_name = col_pair[1].split("_SPLITTER_", 1)[1]
        col_2_changed_name = column_truename_changedname_mapping[col_2_true_name]

        if col_1_true_name == col_2_true_name:
            column_similarity_with_variation_ideal_labels[
                (col_1_true_name, col_2_changed_name)
            ] = "similar"
            if column_similarity_scores[col_pair] > similar_threshold:
                column_similarity_with_variation_predicted_labels[
                    (col_1_true_name, col_2_changed_name)
                ] = "similar"
            else:
                column_similarity_with_variation_predicted_labels[
                    (col_1_true_name, col_2_changed_name)
                ] = "not_similar"
                wrong_preds[
                    (col_1_true_name, col_2_changed_name)
                ] = column_similarity_scores[col_pair]
        else:
            if ideal_infotypes[dataset_name_1].get(
                col_1_true_name, None
            ) and ideal_infotypes[dataset_name_1].get(col_2_true_name, None):
                if (
                    ideal_infotypes[dataset_name_1][col_1_true_name]
                    == ideal_infotypes[dataset_name_1][col_2_true_name]
                ):
                    column_similarity_with_variation_ideal_labels[
                        (col_1_true_name, col_2_changed_name)
                    ] = "similar"
                    if column_similarity_scores[col_pair] > similar_threshold:
                        column_similarity_with_variation_predicted_labels[
                            (col_1_true_name, col_2_changed_name)
                        ] = "similar"
                    else:
                        column_similarity_with_variation_predicted_labels[
                            (col_1_true_name, col_2_changed_name)
                        ] = "not_similar"
                        wrong_preds[
                            (col_1_true_name, col_2_changed_name)
                        ] = column_similarity_scores[col_pair]
                else:
                    column_similarity_with_variation_ideal_labels[
                        (col_1_true_name, col_2_changed_name)
                    ] = "not_similar"
                    if column_similarity_scores[col_pair] > similar_threshold:
                        column_similarity_with_variation_predicted_labels[
                            (col_1_true_name, col_2_changed_name)
                        ] = "similar"
                        wrong_preds[
                            (col_1_true_name, col_2_changed_name)
                        ] = column_similarity_scores[col_pair]
                    else:
                        column_similarity_with_variation_predicted_labels[
                            (col_1_true_name, col_2_changed_name)
                        ] = "not_similar"

    comb = (dataset_name_1, dataset_name_1 + "_LOGICAL_COPY")
    (
        tables_predicted_scores,
        tables_predicted_labels,
        tables_actual_labels,
        columns_predicted_scores,
        columns_predicted_labels,
        columns_actual_labels,
        columns_correct_preds,
        columns_wrong_preds,
    ) = update_scores_and_labels_dicts(
        comb,
        overall_table_similarity_score,
        column_similarity_scores,
        columns_predicted_scores,
        columns_predicted_labels,
        columns_actual_labels,
        columns_correct_preds,
        columns_wrong_preds,
        tables_predicted_scores,
        tables_predicted_labels,
        tables_actual_labels,
    )

with open(
    os.path.join(input_jsons_dir, "column_name_variations_wrong_preds.pkl"), "wb"
) as filename:
    pickle.dump(wrong_preds, filename)
#  Display prediction statistics ###
logger.info("-------Test Statistics-------------")
logger.info(f"Correct predictions (Column Similarity) : {len(columns_correct_preds)}")
logger.info(f"Wrong predictions (Column Similarity) : {len(columns_wrong_preds)}")
logger.info(
    f"Accuracy: {np.round(len(columns_correct_preds) / (len(columns_wrong_preds) + len(columns_correct_preds)), 2)}"
)
get_prediction_statistics(columns_predicted_labels, columns_actual_labels, "columns")
get_prediction_statistics(tables_predicted_labels, tables_actual_labels, "tables")

# Display statistics for testing variations in col_name
logger.info("--Statistics for testing variations in col_name----")
get_prediction_statistics(
    column_similarity_with_variation_predicted_labels,
    column_similarity_with_variation_ideal_labels,
    "columns_with_name_variation",
)


# update json files if required ###
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
    ) as jsonfile:
        json.dump(expected_columns_similarity_scores, jsonfile, indent=4)

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
    ) as jsonfile:
        json.dump(expected_tables_similarity_scores, jsonfile, indent=4)


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
    ) as jsonfile:
        json.dump(expected_tables_similarity_labels, jsonfile, indent=4)


# obtain mappings for unit testing
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
