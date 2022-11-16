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
from datahub_classify.sample_input import input1 as input_dict

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
    print("===============%s=================" % input_data_path)
    data1 = pd.read_csv(os.path.join(input_data_path, "UCI_Credit_Card.csv"))
    data2 = pd.read_csv(os.path.join(input_data_path, "Age2_address1_credit_card3.csv"))
    data3 = pd.read_csv(os.path.join(input_data_path, "list_of_real_usa_addresses.csv"))
    data4 = pd.read_csv(os.path.join(input_data_path, "CardBase.csv"))
    data5 = pd.read_csv(os.path.join(input_data_path, "Credit_Card2.csv"))
    data6 = pd.read_csv(os.path.join(input_data_path, "catalog.csv"))
    data7 = pd.read_csv(os.path.join(input_data_path, "iban.csv"))
    # data8 = pd.read_csv(os.path.join(input_data_path, "USA_cars_datasets.csv"))
    # data9 = pd.read_csv(os.path.join(input_data_path, "email_1.csv"))
    # data10 = pd.read_csv(os.path.join(input_data_path, "email_2.csv"))
    # data11 = pd.read_csv(os.path.join(input_data_path, "email_3.csv"))
    data12 = pd.read_csv(
        os.path.join(input_data_path, "2018-seattle-business-districts.csv")
    )
    data13 = pd.read_csv(os.path.join(input_data_path, "Customer_Segmentation.csv"))
    data14 = pd.read_csv(os.path.join(input_data_path, "application_record.csv"))
    data15 = pd.read_csv(os.path.join(input_data_path, "Airbnb_Open_Data.csv"))
    data16 = pd.read_csv(
        os.path.join(input_data_path, "Book1.xlsx-credit-card-number.csv")
    )
    data17 = pd.read_csv(os.path.join(input_data_path, "Aliases.csv"))
    # data18 = pd.read_csv(os.path.join(input_data_path, "athletes.csv"))
    # data19 = pd.read_csv(os.path.join(input_data_path, "coaches.csv"))
    # data20 = pd.read_csv(os.path.join(input_data_path, "curling_results.csv"))
    data21 = pd.read_csv(os.path.join(input_data_path, "Emails.csv"))
    # data22 = pd.read_csv(os.path.join(input_data_path, "hockey_players_stats.csv"))
    # data23 = pd.read_csv(os.path.join(input_data_path, "hockey_results.csv"))
    # data24 = pd.read_csv(os.path.join(input_data_path, "medals.csv"))
    data25 = pd.read_csv(os.path.join(input_data_path, "Persons.csv"))
    # data26 = pd.read_csv(os.path.join(input_data_path, "technical_officials.csv"))
    data27 = pd.read_csv(os.path.join(input_data_path, "Bachelor_Degree_Majors.csv"))
    data28 = pd.read_csv(os.path.join(input_data_path, "CrabAgePrediction.csv"))
    data29 = pd.read_csv(os.path.join(input_data_path, "Salary_Data.csv"))
    data30 = pd.read_csv(os.path.join(input_data_path, "drug-use-by-age.csv"))
    data31 = pd.read_csv(
        os.path.join(input_data_path, "Book1.xlsx-us-social-security-22-cvs.csv")
    )
    data32 = pd.read_csv(os.path.join(input_data_path, "sample-data.csv"))
    data33 = pd.read_excel(os.path.join(input_data_path, "1-MB-Test.xlsx"))
    data34 = pd.read_csv(os.path.join(input_data_path, "random_ibans.csv"))
    # data35 = pd.read_csv(
    #     os.path.join(input_data_path, "used_cars_data.csv"), nrows=1000
    # )
    data36 = pd.read_csv(os.path.join(input_data_path, "train.csv"), nrows=1000)
    data37 = pd.read_csv(os.path.join(input_data_path, "test.csv"), nrows=1000)
    data38 = pd.read_csv(os.path.join(input_data_path, "vehicles_1.csv"), nrows=1000)
    data39 = pd.read_csv(os.path.join(input_data_path, "vehicles_2.csv"), nrows=1000)
    data40 = pd.read_csv(os.path.join(input_data_path, "vehicles_3.csv"), nrows=1000)
    # data41 = pd.read_csv(
    #     os.path.join(input_data_path, "Dataset-Unicauca-Version2-87Atts_1.csv")
    # )
    # data42 = pd.read_csv(
    #     os.path.join(input_data_path, "Dataset-Unicauca-Version2-87Atts_2.csv")
    # )
    # data43 = pd.read_csv(
    #     os.path.join(input_data_path, "Dataset-Unicauca-Version2-87Atts_3.csv")
    # )
    # data44 = pd.read_csv(
    #     os.path.join(input_data_path, "Dataset-Unicauca-Version2-87Atts_4.csv")
    # )
    # data45 = pd.read_csv(
    #     os.path.join(input_data_path, "Dataset-Unicauca-Version2-87Atts_5.csv")
    # )
    # data46 = pd.read_csv(
    #     os.path.join(input_data_path, "visitor-interests.csv"), nrows=1000
    # )
    # data47 = pd.read_csv(
    #     os.path.join(input_data_path, "Darknet_.csv"), nrows=1000, on_bad_lines="skip"
    # )
    data48 = pd.read_csv(os.path.join(input_data_path, "vehicles_4.csv"))
    data49 = pd.read_csv(os.path.join(input_data_path, "vehicles_5.csv"))
    # data50 = pd.read_csv(
    #     os.path.join(
    #         input_data_path, "Device Report - BU175-VPC2021-03-21_11-00-03.csv"
    #     )
    # )
    # data51 = pd.read_csv(
    #     os.path.join(
    #         input_data_path,
    #         "2021-04-23_honeypot-cloud-digitalocean-geo-1_netflow-extended.csv",
    #     ),
    #     nrows=1000,
    # )
    data52 = pd.read_csv(os.path.join(input_data_path, "ipv6_random_generated.csv"))
    # data53 = pd.read_csv(
    #     os.path.join(input_data_path, "score-banks-updated-sep2022.csv")
    # )
    # data54 = pd.read_csv(os.path.join(input_data_path, "blz-aktuell-xlsx-data.csv"))
    # data55 = pd.read_csv(os.path.join(input_data_path, "automotive_service_data.csv"))
    data56 = pd.read_excel(os.path.join(input_data_path, "US_Driving_License.xlsx"))

    return {
        "data1": data1,
        "data2": data2,
        "data3": data3,
        "data4": data4,
        "data5": data5,
        "data6": data6,
        "data7": data7,
        # "data8": data8,
        # "data9": data9,
        # "data10": data10,
        # "data11": data11,
        "data12": data12,
        "data13": data13,
        "data14": data14,
        "data15": data15,
        "data16": data16,
        "data17": data17,
        # "data18": data18,
        # "data19": data19,
        # "data20": data20,
        "data21": data21,
        # "data22": data22,
        # "data23": data23,
        # "data24": data24,
        "data25": data25,
        # "data26": data26,
        "data27": data27,
        "data28": data28,
        "data29": data29,
        "data30": data30,
        "data31": data31,
        "data32": data32,
        "data33": data33,
        "data34": data34,
        # "data35": data35,
        "data36": data36,
        "data37": data37,
        "data38": data38,
        "data39": data39,
        "data40": data40,
        # "data41": data41,
        # "data42": data42,
        # "data43": data43,
        # "data44": data44,
        # "data45": data45,
        # "data46": data46,
        # "data47": data47,
        "data48": data48,
        "data49": data49,
        # "data50": data50,
        # "data51": data51,
        "data52": data52,
        # "data53": data53,
        # "data54": data54,
        # "data55": data55,
        "data56": data56,
    }


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
