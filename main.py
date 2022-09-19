import pandas as pd
from datetime import datetime

from helper_classes import Metadata, ColumnInfo
from infotype_predictor import predict_infotypes


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
    return [data1, data2, data3, data4, data5, data6, data7, data8, data9,
            data10, data11, data12, data13, data14, data15, data16, data17,
            data18, data19, data20, data21, data22, data23, data24, data25, data26]


def populate_column_info_list(data_list):
    column_info_list = []
    actual_labels = []
    for i, data in enumerate(data_list):
        for col in data.columns:
            fields = {
                'Name': col,
                'Description': f'This column contains name of the {col}',
                'Datatype': 'str'
            }
            metadata = Metadata(fields)
            if len(data[col].dropna()) > 1000:
                values = data[col].dropna().values[:1000]
            else:
                values = data[col].dropna().values
            col_info = ColumnInfo(metadata, values)
            column_info_list.append(col_info)
            actual_labels.append(col)
    # return column_info_list, actual_labels
    return column_info_list


def test_predict_infotype(column_info_list, confidence_threshold, input_dict):
    start_time = datetime.now()
    column_info_pred_list = predict_infotypes(column_info_list, confidence_threshold, input_dict)
    end_time = datetime.now()
    print(len(column_info_pred_list))
    print("total time :", end_time - start_time)

    for col_info in column_info_pred_list:
        print('Column Name: ', col_info.metadata.name)
        print('Sample Values: ', col_info.values[:5])
        if col_info.infotype_proposals is None:
            print(col_info.infotype_proposals)
        else:
            for i in range(len(col_info.infotype_proposals)):
                print(f'Proposed InfoType {i + 1} :', col_info.infotype_proposals[i].infotype)
                print('Overall Confidence: ', col_info.infotype_proposals[i].confidence_level)
                print('Debug Info: ', col_info.infotype_proposals[i].debug_info)
                print('--------------------')
        print("\n================================\n")


def run_test(input_data_path):
    from sample_input import input1 as input_dict
    data_list = get_public_data(input_data_path)
    column_info_list = populate_column_info_list(data_list)
    confidence_threshold = 0.6
    test_predict_infotype(column_info_list, confidence_threshold, input_dict)


if __name__ == '__main__':
    # input_data_dir = "C:\\Glossary_Terms\\datasets\\"
    # input_data_dir = '../../../../../../jupyter/office_project/acryl_glossary_term/dataset/'
    input_data_dir = './'
    run_test(input_data_dir)

