import json
import logging
import os
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from datahub_classify.helper_classes import ColumnInfo, Metadata
from datahub_classify.infotype_predictor import predict_infotypes
from datahub_classify.reference_input import input1 as input_dict

logger = logging.getLogger(__name__)

current_wdr = os.path.dirname(os.path.abspath(__file__))
input_data_dir = os.path.join(current_wdr, "datasets")
input_jsons_dir = os.path.join(current_wdr, "expected_output")
confidence_threshold = 0.6
update_confidence_slabs_json = False

logging_directory = os.path.join(current_wdr, "logs", "logs.log")

infotypes_to_use = [
    "Street_Address",
    "Gender",
    "Credit_Debit_Card_Number",
    "Email_Address",
    "Phone_Number",
    "Full_Name",
    "Age",
    "IBAN",
    "Vehicle_Identification_Number",
    "US_Social_Security_Number",
    "IP_Address_v4",
    "IP_Address_v6",
    "Swift_Code",
    "US_Driving_License_Number",
]


def get_public_data(input_data_path):
    logger.info("===============%s=================" % input_data_path)
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


def populate_column_info_list(public_data_list):
    column_info_list = []
    actual_labels = []
    for i, (dataset_name, data) in enumerate(public_data_list.items()):
        for col in data.columns:
            fields = {
                "Name": col,
                "Description": f"This column contains name of the {col}",
                "Datatype": "str",
                "Dataset_Name": dataset_name,
            }
            metadata = Metadata(fields)
            if len(data[col].dropna()) > 1000:
                values = data[col].dropna().values[:1000]
            else:
                values = data[col].dropna().values
            col_info = ColumnInfo(metadata, values)
            column_info_list.append(col_info)
            actual_labels.append(col)
    return column_info_list


def get_public_data_expected_output(public_data_list, infotypes_to_use):
    with open(os.path.join(input_jsons_dir, "expected_infotypes_IDEAL.json")) as file:
        expected_output_ideal = json.load(file)
    with open(
        os.path.join(input_jsons_dir, "expected_infotypes_UNIT_TESTING.json")
    ) as file:
        expected_output_unit_testing = json.load(file)
    with open(
        os.path.join(input_jsons_dir, "expected_infotypes_confidence_slabs.json")
    ) as file:
        expected_infotypes_confidence_slabs = json.load(file)

    for dataset in public_data_list.keys():
        for col in public_data_list[dataset].columns:
            if (col not in expected_output_ideal[dataset].keys()) or (
                expected_output_ideal[dataset][col] not in infotypes_to_use
            ):
                expected_output_ideal[dataset][col] = "no_infotype"

            if not expected_infotypes_confidence_slabs.get(dataset, None):
                expected_infotypes_confidence_slabs[dataset] = dict()
            if (col not in expected_infotypes_confidence_slabs[dataset].keys()) or (
                expected_output_ideal[dataset][col] not in infotypes_to_use
            ):
                expected_infotypes_confidence_slabs[dataset][col] = 0.0

            if col not in expected_output_unit_testing[dataset].keys():
                expected_output_unit_testing[dataset][col] = "no_infotype"

    return (
        expected_output_ideal,
        expected_output_unit_testing,
        expected_infotypes_confidence_slabs,
    )


def get_best_infotype_pred(
    public_data_list,
    confidence_threshold,
    expected_output_unit_testing,
    update_confidence_slabs_json=False,
):
    with open(
        os.path.join(input_jsons_dir, "expected_infotypes_confidence_slabs.json")
    ) as filename:
        old_confidence_slabs = json.load(filename)
    column_info_list = populate_column_info_list(public_data_list)
    column_info_pred_list = predict_infotypes(
        column_info_list, confidence_threshold, input_dict, infotypes_to_use
    )
    public_data_predicted_infotype: Dict[str, dict] = dict()
    # get_thresholds_for_unit_test = dict()
    public_data_predicted_infotype_confidence: Dict[str, dict] = dict()
    for dataset in public_data_list.keys():
        public_data_predicted_infotype[dataset] = dict()
        if not old_confidence_slabs.get(dataset):
            old_confidence_slabs[dataset] = dict()
        public_data_predicted_infotype_confidence[dataset] = dict()
        for col in public_data_list[dataset].columns:
            for col_info in column_info_pred_list:
                if (
                    col_info.metadata.name == col
                    and col_info.metadata.dataset_name == dataset
                ):
                    public_data_predicted_infotype[dataset][col] = "no_infotype"
                    if (
                        col_info.infotype_proposals
                        and len(col_info.infotype_proposals) > 0
                    ):
                        highest_confidence_level: float = 0
                        infotype_assigned = None
                        for i in range(len(col_info.infotype_proposals)):
                            if (
                                col_info.infotype_proposals[i].confidence_level
                                > highest_confidence_level
                            ):
                                # get_thresholds_for_unit_test[dataset][col] = col_info.infotype_proposals[i].confidence_level
                                highest_confidence_level = col_info.infotype_proposals[
                                    i
                                ].confidence_level
                                infotype_assigned = col_info.infotype_proposals[
                                    i
                                ].infotype
                        if not old_confidence_slabs[dataset].get(col):
                            old_confidence_slabs[dataset][col] = (
                                np.floor(highest_confidence_level * 10) / 10
                            )
                        public_data_predicted_infotype[dataset][col] = infotype_assigned
                        public_data_predicted_infotype_confidence[dataset][
                            col
                        ] = highest_confidence_level

                        # TODO: what is the use of following condition?
                        if expected_output_unit_testing[dataset][col] not in (
                            infotypes_to_use + ["no_infotype"]
                        ):
                            expected_output_unit_testing[dataset][
                                col
                            ] = infotype_assigned
                    else:
                        if expected_output_unit_testing[dataset][col] not in (
                            infotypes_to_use + ["no_infotype"]
                        ):
                            expected_output_unit_testing[dataset][col] = "no_infotype"
                        public_data_predicted_infotype_confidence[dataset][col] = 0.0
    if update_confidence_slabs_json:
        with open(
            os.path.join(input_jsons_dir, "expected_infotypes_confidence_slabs.json"),
            "w",
        ) as filename:
            json.dump(old_confidence_slabs, filename, indent=4)
    return (
        public_data_predicted_infotype,
        expected_output_unit_testing,
        public_data_predicted_infotype_confidence,
    )


def get_pred_exp_infotype_mapping(
    public_data_predicted_infotype,
    public_data_expected_infotype,
    expected_infotypes_confidence_slabs,
    public_data_predicted_infotype_confidence,
):
    mapping = []
    for dataset in public_data_predicted_infotype.keys():
        for col in public_data_predicted_infotype[dataset].keys():
            mapping.append(
                (
                    dataset,
                    col,
                    public_data_predicted_infotype[dataset][col],
                    public_data_expected_infotype[dataset][col],
                    public_data_predicted_infotype_confidence[dataset][col],
                    expected_infotypes_confidence_slabs[dataset][col],
                )
            )
    return mapping


def get_prediction_statistics(mapping, infotypes_to_use, confidence_threshold):
    infotypes_to_use.append("no_infotype")
    all_infotypes = infotypes_to_use
    y_true = [s[3] for s in mapping]
    y_pred = [s[2] for s in mapping]
    prediction_stats = pd.DataFrame()
    prediction_stats["Infotype"] = all_infotypes
    prediction_stats["Precision"] = np.round(
        precision_score(y_true, y_pred, average=None, labels=all_infotypes), 2
    )
    prediction_stats["Recall"] = np.round(
        recall_score(y_true, y_pred, average=None, labels=all_infotypes), 2
    )
    df_confusion_matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=all_infotypes),
        columns=[info + "_predicted" for info in all_infotypes],
        index=[info + "_actual" for info in all_infotypes],
    )
    logger.info("*************Prediction Statistics***************************")
    logger.info(prediction_stats)
    logger.info("********************")
    logger.info(df_confusion_matrix)
    prediction_stats.to_csv(
        f"Prediction_statistics_{confidence_threshold}.csv", index=False
    )
    df_confusion_matrix.to_csv(f"confusion_matrix_{confidence_threshold}.csv")


# TODO: think of adding 'if __name__ == “main”' block for following executable code
# if __name__ == '__main__':
logger.info("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
logger.info("--------------------STARTING RUN--------------------  ")
logger.info("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
logger.info(
    f"Start Time --->  {datetime.now()}",
)

public_data_list = get_public_data(input_data_dir)
(
    expected_output_ideal,
    expected_output_unit_testing,
    expected_infotypes_confidence_slabs,
) = get_public_data_expected_output(public_data_list, infotypes_to_use)
(
    public_data_predicted_infotype,
    expected_output_unit_testing,
    public_data_predicted_infotype_confidence,
) = get_best_infotype_pred(
    public_data_list,
    confidence_threshold,
    expected_output_unit_testing,
    update_confidence_slabs_json,
)
infotype_mapping_ideal = get_pred_exp_infotype_mapping(
    public_data_predicted_infotype,
    expected_output_ideal,
    expected_infotypes_confidence_slabs,
    public_data_predicted_infotype_confidence,
)
infotype_mapping_unit_testing = get_pred_exp_infotype_mapping(
    public_data_predicted_infotype,
    expected_output_unit_testing,
    expected_infotypes_confidence_slabs,
    public_data_predicted_infotype_confidence,
)
get_prediction_statistics(
    infotype_mapping_ideal, infotypes_to_use, confidence_threshold
)


@pytest.mark.parametrize(
    "dataset_name,column_name, predicted_output, expected_output,"
    "predicted_output_confidence, expected_confidence_slab",
    [(a, b, c, d, e, f) for a, b, c, d, e, f in infotype_mapping_unit_testing],
)
def test_public_datasets(
    dataset_name,
    column_name,
    predicted_output,
    expected_output,
    predicted_output_confidence,
    expected_confidence_slab,
):
    assert (
        predicted_output == expected_output
    ), f"Test1 failed for column '{column_name}' in {dataset_name}"
    assert predicted_output_confidence >= expected_confidence_slab, (
        f"Test2 failed for column '{column_name}' in " f"{dataset_name}"
    )
