import json
import logging
import os
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import pickle

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
update_expected_similarity_scores_UNIT_TESTING = False
quick_test = False
save_predictions = False

def get_public_data(input_data_path, run_quick_test= False):
    logger.info(f"==============={input_data_path}=================")
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

    if run_quick_test:
        return {
            "data1": data1,
            "data2": data2,
            "data3": data3
        }
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

dataset_name_mapping = {
    "data1": "UCI_Credit_Card",
    "data2": "Age2_address1_credit_card3",
    "data3": "list_of_real_usa_addresses",
    "data4": "CardBase",
    "data5": "Credit_Card2",
    "data6": "catalog",
    "data7": "iban",
    "data8": "USA_cars_datasets",
    "data9": "email_1",
    "data10": "email_2",
    "data11": "email_3",
    "data12": "2018-seattle-business-districts",
    "data13": "Customer_Segmentation",
    "data14": "application_record",
    "data15": "Airbnb_Open_Data",
    "data16": "Book1.xlsx-credit-card-number",
    "data17": "Aliases",
    "data18": "athletes",
    "data19": "coaches",
    "data20": "curling_results",
    "data21": "Emails",
    "data22": "hockey_players_stats",
    "data23": "hockey_results",
    "data24": "medals",
    "data25": "Persons",
    "data26": "technical_officials",
    "data27": "Bachelor_Degree_Majors",
    "data28": "CrabAgePrediction",
    "data29": "Salary_Data",
    "data30": "drug-use-by-age",
    "data31": "Book1.xlsx-us-social-security-22-cvs",
    "data32": "sample-data",
    "data33": "1-MB-Test",
    "data34": "random_ibans",
    "data35": "used_cars_data",
    "data36": "train",
    "data37": "test",
    "data38": "vehicles_1",
    "data39": "vehicles_2",
    "data40": "vehicles_3",
    "data41": "Dataset-Unicauca-Version2-87Atts_1",
    "data42": "Dataset-Unicauca-Version2-87Atts_2",
    "data43": "Dataset-Unicauca-Version2-87Atts_3",
    "data44": "Dataset-Unicauca-Version2-87Atts_4",
    "data45": "Dataset-Unicauca-Version2-87Atts_5",
    "data46": "visitor-interests",
    "data47": "Darknet_",
    "data48": "vehicles_4",
    "data49": "vehicles_5",
    "data50": "Device Report - BU175-VPC2021-03-21_11-00-03",
    "data51": "2021-04-23_honeypot-cloud-digitalocean-geo-1_netflow-extended",
    "data52": "ipv6_random_generated",
    "data53": "score-banks-updated-sep2022",
    "data54": "blz-aktuell-xlsx-data",
    "data55": "automotive_service_data",
    "data56": "US_Driving_License"
}

platforms = ["A", "B", "C", "D", "E"]


def populate_tableinfo_object(dataset_key):
    table_meta_info = {
        "Name": dataset_name_mapping[dataset_key],
        "Description": f"This table contains description of {dataset_name_mapping[dataset_key]}",
        "Platform": platforms[np.random.randint(0, 5)],
        "Table_Id": dataset_key,
    }
    col_infos = []
    for col in public_data_list[dataset_key].columns:
        fields = {
            "Name": col,
            "Description": f" {col}",
            "Datatype": public_data_list[dataset_key][col].dropna().dtype,
            "Dataset_Name": dataset_key,
            "Column_Id": dataset_key + "_" + col,
        }
        metadata_col = ColumnMetadata(fields)
        # parent_cols = list()
        col_info = ColumnInfo(metadata_col)
        col_infos.append(col_info)

    metadata_table = TableMetadata(table_meta_info)
    # parent_tables = list()
    table_info = TableInfo(metadata_table, col_infos)
    return table_info

def populate_similar_tableinfo_object(dataset_key):
    df = public_data_list[dataset_key].copy()
    random_df_key = "data" + str(np.random.randint(1, len(dataset_name_mapping)))
    while not (dataset_name_mapping.get(random_df_key, None)) or (
        random_df_key == dataset_key
    ):
        random_df_key = "data" + str(np.random.randint(1, len(dataset_name_mapping)))
    random_df = public_data_list[random_df_key].copy()

    df.columns = [dataset_key + "_" + col for col in df.columns]
    random_df.columns = [random_df_key + "_" + col for col in random_df.columns]
    second_df = pd.concat([df, random_df], axis=1)
    cols_to_keep = list(df.columns) + list(random_df.columns[:2])
    second_df = second_df[cols_to_keep]
    table_meta_info = {
        "Name": dataset_name_mapping[dataset_key] + "_v2",
        "Description": f" {dataset_name_mapping[dataset_key]}",
        "Platform": platforms[np.random.randint(0, 5)],
        "Table_Id": dataset_key + "_" + random_df_key,
    }
    col_infos = []
    for col in second_df.columns:
        fields = {
            "Name": col.split("_", 1)[1],
            "Description": f'{col.split("_", 1)[1]}',
            "Datatype": second_df[col].dropna().dtype,
            "Dataset_Name": dataset_key + "_" + random_df_key,
            "Column_Id": col,
        }
        metadata_col = ColumnMetadata(fields)
        parent_cols = [dataset_key + "_" + col if col in df.columns else None]
        col_info = ColumnInfo(metadata_col, parent_cols)
        col_infos.append(col_info)
    metadata_table = TableMetadata(table_meta_info)
    parent_tables = [dataset_key]
    table_info = TableInfo(metadata_table, parent_tables, col_infos)
    return table_info

def load_expected_similarity_json(input_jsons_path):
    with open(
            os.path.join(input_jsons_path, "expected_similarity_scores_UNIT_TESTING.json")) as filename:
        expected_similarity_scores_unit_testing = json.load(filename)
    return expected_similarity_scores_unit_testing


def get_predicted_expected_similarity_scores_mapping(predicted_similarity_scores_unit_testing,
                                            predicted_similarity_labels_unit_testing,
                                            expected_similarity_scores_unit_testing):
    mapping = []
    for column_pair in predicted_similarity_scores_unit_testing.keys():
        key_ = column_pair[0] + "-" + column_pair[1]
        if expected_similarity_scores_unit_testing.get(key_, None):
            if expected_similarity_scores_unit_testing[key_] >= similar_threshold:
                expected_similarity_label_unit_testing = "similar"
            else:
                expected_similarity_label_unit_testing = "not_similar"
            mapping.append((column_pair[0],
                            column_pair[1],
                           predicted_similarity_scores_unit_testing[column_pair],
                           predicted_similarity_labels_unit_testing[column_pair],
                           expected_similarity_scores_unit_testing[key_],
                           expected_similarity_label_unit_testing))
    return mapping


logger.info("----------Starting Testing-----------")
public_data_list = get_public_data(input_data_dir, quick_test)
data_combinations = combinations(public_data_list.keys(), 2)
correct_preds = []
wrong_preds = []
predicted_labels = dict()
predicted_scores = dict()
for comb in data_combinations:
    logger.info(f"Processing_combination: {comb}")
    dataset_key_1 = comb[0]
    dataset_key_2 = comb[1]
    table_info_1 = populate_tableinfo_object(dataset_key=dataset_key_1)
    table_info_2 = populate_tableinfo_object(dataset_key=dataset_key_2)
    overall_table_similarity_score, column_similarity_scores = check_similarity(
        table_info_1, table_info_2
    )
    for col_pair in column_similarity_scores.keys():
        predicted_scores[col_pair] = column_similarity_scores[col_pair]
        col_1 = col_pair[0].split("_", 1)[1]
        col_2 = col_pair[1].split("_", 1)[1]
        if ideal_infotypes[dataset_key_1].get(col_1, None) and ideal_infotypes[
            dataset_key_2
        ].get(col_2, None):
            if (
                ideal_infotypes[dataset_key_1][col_1]
                == ideal_infotypes[dataset_key_2][col_2]
            ):
                if column_similarity_scores[col_pair] >= similar_threshold:
                    predicted_labels[col_pair] = "similar"
                    correct_preds.append((col_pair, column_similarity_scores[col_pair]))
                else:
                    predicted_labels[col_pair] = "not_similar"
                    wrong_preds.append((col_pair, column_similarity_scores[col_pair]))
            else:
                if column_similarity_scores[col_pair] >= similar_threshold:
                    predicted_labels[col_pair] = "similar"
                    wrong_preds.append((col_pair, column_similarity_scores[col_pair]))
                else:
                    predicted_labels[col_pair] = "not_similar"
                    correct_preds.append((col_pair, column_similarity_scores[col_pair]))
logger.info("-------Test Statistics-------------")
logger.info(f"Correct predictions: {len(correct_preds)}")
logger.info(f"Wrong predictions: {len(wrong_preds)}")
logger.info(f"Accuracy: {np.round(len(correct_preds) / (len(wrong_preds) + len(correct_preds)), 2)}")

if save_predictions:
    with open(os.path.join(input_jsons_dir,"wrong_predictions.pkl"),"wb") as file:
        pickle.dump(wrong_preds, file)
    with open(os.path.join(input_jsons_dir,"correct_predictions.pkl"),"wb") as file:
        pickle.dump(correct_preds, file)


if update_expected_similarity_scores_UNIT_TESTING:
    expected_similarity_scores = {}
    for key in predicted_scores.keys():
        col_1 = key[0].split("_", 1)[1]
        dataset_key_1 = key[0].split("_", 1)[0]
        col_2 = key[1].split("_", 1)[1]
        dataset_key_2 = key[1].split("_", 1)[0]
        if ideal_infotypes[dataset_key_1].get(col_1, None) and ideal_infotypes[
            dataset_key_2
        ].get(col_2, None):
            expected_similarity_scores[key[0] +"-" + key[1]] = predicted_scores[key]
    logger.info("Updating expected_similarity_scores_UNIT_TESTING json..")
    with open(os.path.join(input_jsons_dir, "expected_similarity_scores_UNIT_TESTING.json"),"w") as filename:
            json.dump(expected_similarity_scores, filename, indent = 4)

expected_similarity_scores_unit_testing = load_expected_similarity_json(input_jsons_dir)
similarity_mapping_unit_testing = get_predicted_expected_similarity_scores_mapping(predicted_scores,
                                                                          predicted_labels,
                                                                          expected_similarity_scores_unit_testing)

@pytest.mark.parametrize(
    "col_id_1, col_id_2, predicted_score, predicted_label, expected_score, expected_label",
    [(a, b, c, d, e, f) for a, b, c, d, e, f in similarity_mapping_unit_testing],
)
def test_similarity_public_datasets(
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
    assert predicted_score >= np.floor(expected_score * 10) / 10, (
        f"Test2 failed for column pair: '{(col_id_1, col_id_2)}'"
    )


