import os
import sys
import inspect
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import json
import pytest

from main import check_predict_infotype
from helper_classes import Metadata, ColumnInfo
from sample_input import input1 as input_dict
from supported_infotypes import infotypes_to_use


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

input_data_dir = './datasets/'
confidence_threshold = 0.6


def get_public_data(input_data_path):
    # TODO: Iterate over the directory and load all datasets
    data1 = pd.read_csv(input_data_path + "UCI_Credit_Card.csv")
    data2 = pd.read_csv(input_data_path + "Age2_address1_credit_card3.csv")
    data3 = pd.read_csv(input_data_path + "list_of_real_usa_addresses.csv")
    data4 = pd.read_csv(input_data_path + "CardBase.csv")
    data5 = pd.read_csv(input_data_path + "Credit_Card2.csv")
    data6 = pd.read_csv(input_data_path + "catalog.csv")
    data7 = pd.read_csv(input_data_path + "iban.csv")
    data8 = pd.read_csv(input_data_path + "USA_cars_datasets.csv")
    data9 = pd.read_csv(input_data_path + "email_1.csv")
    data10 = pd.read_csv(input_data_path + "email_2.csv")
    data11 = pd.read_csv(input_data_path + "email_3.csv")
    data12 = pd.read_csv(input_data_path + "2018-seattle-business-districts.csv")
    data13 = pd.read_csv(input_data_path + "Customer Segmentation.csv")
    data14 = pd.read_csv(input_data_path + "application_record.csv")
    data15 = pd.read_csv(input_data_path + "Airbnb_Open_Data.csv")
    data16 = pd.read_csv(input_data_path + "Book1.xlsx-credit-card-number.csv")
    data17 = pd.read_csv(input_data_path + "Aliases.csv")
    data18 = pd.read_csv(input_data_path + "athletes.csv")
    data19 = pd.read_csv(input_data_path + "coaches.csv")
    data20 = pd.read_csv(input_data_path + "curling_results.csv")
    data21 = pd.read_csv(input_data_path + "Emails.csv")
    data22 = pd.read_csv(input_data_path + "hockey_players_stats.csv")
    data23 = pd.read_csv(input_data_path + "hockey_results.csv")
    data24 = pd.read_csv(input_data_path + "medals.csv")
    data25 = pd.read_csv(input_data_path + "Persons.csv")
    data26 = pd.read_csv(input_data_path + "technical_officials.csv")
    return {'data1': data1, 'data2': data2, 'data3': data3, 'data4': data4, 'data5': data5,
            'data6': data6, 'data7': data7, 'data8': data8, 'data9': data9, 'data10': data10,
            'data11': data11, 'data12': data12, 'data13': data13, 'data14': data14, 'data15': data15,
            'data16': data16, 'data17': data17, 'data18': data18, 'data19': data19, 'data20': data20,
            'data21': data21, 'data22': data22, 'data23': data23, 'data24': data24, 'data25': data25,
            'data26': data26}


def populate_column_info_list(public_data_list):
    column_info_list = []
    actual_labels = []
    for i, (dataset_name, data) in enumerate(public_data_list.items()):
        for col in data.columns:
            fields = {
                'Name': col,
                'Description': f'This column contains name of the {col}',
                'Datatype': 'str',
                'Dataset_Name': dataset_name
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
    # TODO: use the absolute path
    # TODO: following code will not work if I run this code from parent directory instead of test dir
    # TODO: from readability perspective, format all three json files (may use some online json formatter)
    with open("expected_output/expected_infotypes_IDEAL.json") as file:
        expected_output_ideal = json.load(file)
    with open("expected_output/expected_infotypes_UNIT_TESTING.json") as file:
        expected_output_unit_testing = json.load(file)
    with open("expected_output/expected_infotypes_confidence_slabs.json") as file:
        expected_infotypes_confidence_slabs = json.load(file)

    for dataset in public_data_list.keys():
        for col in public_data_list[dataset].columns:

            # TODO: can we combine following two conditions with or instead of two if statement?
            # TODO: "e.g. if cond1 or cond2:"
            # TODO: why are we using "no_infotype" string instead of None?
            if col not in expected_output_ideal[dataset].keys():
                expected_output_ideal[dataset][col] = "no_infotype"
            if expected_output_ideal[dataset][col] not in infotypes_to_use:
                expected_output_ideal[dataset][col] = "no_infotype"

            # TODO: can we combine following two conditions with or instead of two if statement?
            if col not in expected_infotypes_confidence_slabs[dataset].keys():
                expected_infotypes_confidence_slabs[dataset][col] = 0.0
            if expected_infotypes_confidence_slabs[dataset][col] not in infotypes_to_use:
                expected_infotypes_confidence_slabs[dataset][col] = 0.0

            if col not in expected_output_unit_testing[dataset].keys():
                expected_output_unit_testing[dataset][col] = "no_infotype"

    return expected_output_ideal, expected_output_unit_testing, expected_infotypes_confidence_slabs


def get_best_infotype_pred(public_data_list, confidence_threshold, expected_output_unit_testing):
    column_info_list = populate_column_info_list(public_data_list)
    # TODO: think of directly calling "predict_infotypes()" function if print statements are not required
    column_info_pred_list = check_predict_infotype(column_info_list, confidence_threshold, input_dict)
    public_data_predicted_infotype = dict()
    # get_thresholds_for_unit_test = dict()
    public_data_predicted_infotype_confidence = dict()
    for dataset in public_data_list.keys():
        public_data_predicted_infotype[dataset] = dict()
        # get_thresholds_for_unit_test[dataset] = dict()
        public_data_predicted_infotype_confidence[dataset] = dict()
        for col in public_data_list[dataset].columns:
            for col_info in column_info_pred_list:
                if col_info.metadata.name == col and col_info.metadata.dataset_name == dataset:
                    public_data_predicted_infotype[dataset][col] = "no_infotype"
                    if len(col_info.infotype_proposals) > 0:
                        highest_confidence_level = 0
                        for i in range(len(col_info.infotype_proposals)):
                            if col_info.infotype_proposals[i].confidence_level > highest_confidence_level:
                                # TODO: highest confidence value and associated infotype assignment in the dict can be done outside of the loop
                                # TODO: just initialize highest_confidence_level and infotype variable in the for loop
                                public_data_predicted_infotype[dataset][col] = col_info.infotype_proposals[i].infotype
                                # get_thresholds_for_unit_test[dataset][col] = col_info.infotype_proposals[i].confidence_level
                                public_data_predicted_infotype_confidence[dataset][col] = col_info.infotype_proposals[i].confidence_level
                                highest_confidence_level = col_info.infotype_proposals[i].confidence_level
                                # TODO: what is the use of following condition?
                                if expected_output_unit_testing[dataset][col] not in (
                                        infotypes_to_use + ["no_infotype"]):
                                    expected_output_unit_testing[dataset][col] = col_info.infotype_proposals[i].infotype
                    else:
                        if expected_output_unit_testing[dataset][col] not in (infotypes_to_use + ["no_infotype"]):
                            expected_output_unit_testing[dataset][col] = "no_infotype"
                        public_data_predicted_infotype_confidence[dataset][col] = 0.0
    # with open("infotype_threholds.json", "w") as filename:
    #     json.dump(get_thresholds_for_unit_test, filename)
    return public_data_predicted_infotype, expected_output_unit_testing, public_data_predicted_infotype_confidence


def get_pred_exp_infotype_mapping(public_data_predicted_infotype, public_data_expected_infotype,
                                  expected_infotypes_confidence_slabs, public_data_predicted_infotype_confidence):
    mapping = []
    for dataset in public_data_predicted_infotype.keys():
        for col in public_data_predicted_infotype[dataset].keys():
            # TODO: shall we have same order of infotype and confidence just for uniform sequence
            # TODO: (dataset, col, pred_infotype, exp_infotype, pred_confidence, exp_confidence)
            mapping.append((dataset, col, public_data_predicted_infotype[dataset][col],
                            public_data_expected_infotype[dataset][col],
                            expected_infotypes_confidence_slabs[dataset][col],
                            public_data_predicted_infotype_confidence[dataset][col]))
    return mapping


def get_prediction_statistics(mapping, infotypes_to_use, confidence_threshold):
    infotypes_to_use.append("no_infotype")
    all_infotypes = infotypes_to_use
    prediction_stats = pd.DataFrame()
    prediction_stats["InfoType_Name"] = all_infotypes
    prediction_stats["TP"] = 0
    prediction_stats["FP"] = 0
    prediction_stats["TN"] = 0
    prediction_stats["FN"] = 0
    # TODO: can we use sklearn functionality to get precision/recall instead of following code?
    for data_name, col_name, pred_val, true_val, _, _ in mapping:
        if pred_val == true_val:
            current_tp_value = prediction_stats.loc[prediction_stats["InfoType_Name"] == true_val, "TP"].values
            current_tp_value += 1
            prediction_stats.loc[prediction_stats["InfoType_Name"] == true_val, "TP"] = current_tp_value
            true_neg_infotypes = [infotype for infotype in all_infotypes if infotype != true_val]
            for infotype in true_neg_infotypes:
                current_tn_value = prediction_stats.loc[prediction_stats["InfoType_Name"] == infotype, "TN"].values
                current_tn_value += 1
                prediction_stats.loc[prediction_stats["InfoType_Name"] == infotype, "TN"] = current_tn_value
        elif pred_val != true_val:
            current_fn_value = prediction_stats.loc[prediction_stats["InfoType_Name"] == true_val, "FN"].values
            current_fn_value += 1
            prediction_stats.loc[prediction_stats["InfoType_Name"] == true_val, "FN"] = current_fn_value
            current_fp_value = prediction_stats.loc[prediction_stats["InfoType_Name"] == pred_val, "FP"].values
            current_fp_value += 1
            prediction_stats.loc[prediction_stats["InfoType_Name"] == pred_val, "FP"] = current_fp_value
            # TODO: don't we need to consider infotypes other than pred_val and true_val for TN case?
    prediction_stats["Precision"] = np.round(prediction_stats["TP"] / (prediction_stats["TP"] + prediction_stats["FP"]),
                                             2)
    prediction_stats["Recall"] = np.round(prediction_stats["TP"] / (prediction_stats["TP"] + prediction_stats["FN"]), 2)
    y_true = [s[3] for s in mapping]
    y_pred = [s[2] for s in mapping]
    df_confusion_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=all_infotypes),
                                       columns=[info + "_predicted" for info in all_infotypes],
                                       index=[info + "_actual" for info in all_infotypes])

    print("*************Prediction Statistics***************************")
    print(prediction_stats)
    print("********************")
    print(df_confusion_matrix)
    prediction_stats.to_csv(f"Prediction_statistics_{confidence_threshold}.csv", index=False)
    df_confusion_matrix.to_csv(f"confusion_matrix_{confidence_threshold}.csv")


# TODO: think of adding 'if __name__ == “main”' block for following executable code
public_data_list = get_public_data(input_data_dir)
expected_output_ideal, expected_output_unit_testing, expected_infotypes_confidence_slabs = \
                                get_public_data_expected_output(public_data_list, infotypes_to_use)
public_data_predicted_infotype, expected_output_unit_testing, public_data_predicted_infotype_confidence =\
                                                                                      get_best_infotype_pred(public_data_list,
                                                                                      confidence_threshold,
                                                                                      expected_output_unit_testing)
infotype_mapping_ideal = get_pred_exp_infotype_mapping(public_data_predicted_infotype,
                                                       expected_output_ideal,
                                                       expected_infotypes_confidence_slabs,
                                                       public_data_predicted_infotype_confidence)
infotype_mapping_unit_testing = get_pred_exp_infotype_mapping(public_data_predicted_infotype,
                                                              expected_output_unit_testing,
                                                              expected_infotypes_confidence_slabs,
                                                              public_data_predicted_infotype_confidence)

get_prediction_statistics(infotype_mapping_ideal, infotypes_to_use, confidence_threshold)


@pytest.mark.parametrize("dataset_name,column_name, predicted_output, expected_output,"
                         "expected_confidence_slab, predicted_output_confidence",
                         [(a, b, c, d, e, f) for a, b, c, d, e, f in infotype_mapping_unit_testing])
def test_public_datasets(dataset_name, column_name, predicted_output, expected_output,
                         expected_confidence_slab,predicted_output_confidence):
    assert predicted_output == expected_output, f"Test1 failed for column '{column_name}' in {dataset_name}"
    assert predicted_output_confidence >= expected_confidence_slab, f"Test2 failed for column '{column_name}' in " \
                                                                    f"{dataset_name}"
