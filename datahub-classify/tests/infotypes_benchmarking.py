import os
import time

import pandas as pd

from datahub_classify.helper_classes import ColumnInfo, Metadata
from datahub_classify.infotype_predictor import predict_infotypes
from datahub_classify.reference_input import input1 as input_dict

NUM_ROWS = 1000
current_wdr = os.path.dirname(os.path.abspath(__file__))
input_data_dir = os.path.join(current_wdr, "datasets")
confidence_threshold = 0.6
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


def get_predictions(public_data_list):
    result_df = pd.DataFrame()
    for i, (dataset_name, data) in enumerate(public_data_list.items()):
        print(f"================ Processing - {dataset_name} =========================")
        result_dict = {}
        data = data.head(NUM_ROWS)
        result_dict["dataset_name"] = dataset_name
        result_dict["num_rows"] = data.shape[0]
        result_dict["num_cols"] = data.shape[1]

        column_info_list = populate_column_info_list({dataset_name: data})
        start_time = time.time()
        _ = predict_infotypes(
            column_info_list, confidence_threshold, input_dict, infotypes_to_use
        )
        end_time = time.time()
        result_dict["execution_time"] = end_time - start_time
        result_df = result_df.append(result_dict, ignore_index=True)
    return result_df


datasets = get_public_data(input_data_dir)
result = get_predictions(datasets)
result.to_csv(
    f"datahub_classify_execution_time_rows_{NUM_ROWS}.csv", header=True, index=False
)
print(result)
print("======================")
print(pd.read_csv(f"datahub_classify_execution_time_rows_{NUM_ROWS}.csv"))
