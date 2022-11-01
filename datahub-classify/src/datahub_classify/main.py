import logging
import os
from datetime import datetime

import pandas as pd

from datahub_classify.helper_classes import ColumnInfo, Metadata
from datahub_classify.infotype_predictor import predict_infotypes

logger = logging.getLogger(__name__)

current_dir = os.getcwd()
logging_directory = current_dir + "/logs/logs.log"


def get_public_data(input_data_path):
    data1 = pd.read_csv(input_data_path + "UCI_Credit_Card.csv")
    data2 = pd.read_csv(input_data_path + "Age2_address1_credit_card3.csv")
    data3 = pd.read_csv(input_data_path + "list_of_real_usa_addresses.csv")
    data4 = pd.read_csv(input_data_path + "CardBase.csv")
    data5 = pd.read_csv(input_data_path + "Credit_Card2.csv")
    data6 = pd.read_csv(input_data_path + "catalog.csv")
    data7 = pd.read_csv(input_data_path + "iban.csv")
    data12 = pd.read_csv(input_data_path + "2018-seattle-business-districts.csv")
    data13 = pd.read_csv(input_data_path + "Customer Segmentation.csv")
    data14 = pd.read_csv(input_data_path + "application_record.csv")
    data15 = pd.read_csv(input_data_path + "Airbnb_Open_Data.csv")
    data16 = pd.read_csv(input_data_path + "Book1.xlsx-credit-card-number.csv")
    data17 = pd.read_csv(input_data_path + "Aliases.csv")
    data21 = pd.read_csv(input_data_path + "Emails.csv")
    data25 = pd.read_csv(input_data_path + "Persons.csv")
    data27 = pd.read_csv(input_data_path + "Bachelor_Degree_Majors.csv")
    data28 = pd.read_csv(input_data_path + "CrabAgePrediction.csv")
    data29 = pd.read_csv(input_data_path + "Salary_Data.csv")
    data30 = pd.read_csv(input_data_path + "drug-use-by-age.csv")
    return {
        "data1": data1,
        "data2": data2,
        "data3": data3,
        "data4": data4,
        "data5": data5,
        "data6": data6,
        "data7": data7,
        "data12": data12,
        "data13": data13,
        "data14": data14,
        "data15": data15,
        "data16": data16,
        "data17": data17,
        "data21": data21,
        "data25": data25,
        "data27": data27,
        "data28": data28,
        "data29": data29,
        "data30": data30,
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


def check_predict_infotype(column_info_list, confidence_threshold, input_dict):
    column_info_pred_list = predict_infotypes(
        column_info_list, confidence_threshold, input_dict
    )
    for col_info in column_info_pred_list:
        logger.debug("Column Name: ", col_info.metadata.name)
        logger.debug("Sample Values: ", col_info.values[:5])
        if col_info.infotype_proposals is None:
            logger.debug(col_info.infotype_proposals)
        else:
            for i in range(len(col_info.infotype_proposals)):
                logger.debug(
                    f"Proposed InfoType {i + 1} :",
                    col_info.infotype_proposals[i].infotype,
                )
                logger.debug(
                    "Overall Confidence: ",
                    col_info.infotype_proposals[i].confidence_level,
                )
                logger.debug("Debug Info: ", col_info.infotype_proposals[i].debug_info)
                logger.debug("--------------------")
        logger.debug("\n================================\n")
    return column_info_pred_list


def run_test(input_data_path):
    from sample_input import input1 as input_dict

    data_list = get_public_data(input_data_path)
    column_info_list = populate_column_info_list(data_list)
    confidence_threshold = 0.6
    check_predict_infotype(column_info_list, confidence_threshold, input_dict)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info("--------------------STARTING RUN--------------------  ")
    logger.info("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info(
        f"Start Time --->  {datetime.now()}",
    )
    # input_data_dir = "C:\\Glossary_Terms_Git\\datahub-classify\\test\\datasets\\"
    # input_data_dir = '../../../../../../jupyter/office_project/acryl_glossary_term/dataset/'
    input_data_dir = "./"
    run_test(input_data_dir)
