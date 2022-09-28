from main import  check_predict_infotype
from helper_classes import Metadata, ColumnInfo
import pytest
from sample_input import input1 as input_dict
from supported_infotypes import infotypes_to_use
import pandas as pd
from sklearn.metrics import confusion_matrix

input_data_dir = "C:\\Glossary_Terms\\datasets\\"
# input_data_dir = '../../../../../../jupyter/office_project/acryl_glossary_term/dataset/'
# input_data_dir = './'

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
    return {'data1':data1,'data2':data2, 'data3': data3, 'data4': data4, 'data5': data5,
            'data6': data6,'data7': data7,'data8': data8,'data9': data9,'data10': data10,
            'data11': data11,'data12': data12,'data13': data13,'data14': data14,'data15': data15,
            'data16': data16,'data17': data17,'data18': data18,'data19': data19,'data20': data20,
            'data21': data21,'data22': data22,'data23': data23,'data24': data24,'data25': data25,
            'data26': data26}


def populate_column_info_list(public_data_list):
    column_info_list = []
    actual_labels = []
    for i, (dataset_name,data) in enumerate(public_data_list.items()):
        for col in data.columns:
            fields = {
                'Name': col,
                'Description': f'This column contains name of the {col}',
                'Datatype': 'str',
            }
            metadata = Metadata(fields)
            metadata.dataset_name = dataset_name
            if len(data[col].dropna()) > 1000:
                values = data[col].dropna().values[:1000]
            else:
                values = data[col].dropna().values
            col_info = ColumnInfo(metadata, values)
            column_info_list.append(col_info)
            actual_labels.append(col)
    return column_info_list


def get_public_data_expected_output(public_data_list, infotypes_to_use):
    expected_output = dict()
    expected_output['data1'] = {'SEX': 'Gender', 'AGE': 'Age'}
    expected_output['data2'] = {'Person': 'Full_Name', 'Current Age': 'Age', 'Retirement Age': 'Age',
                                'Gender': 'Gender', 'Address': 'Street_Address'}
    expected_output['data3'] = {'address':'Street_Address'}
    expected_output['data4'] = {'Card_Number':'Credit_Card_Number'}
    expected_output['data5'] = {'Card Number':'Credit_Card_Number'}
    expected_output['data6'] = {}
    expected_output['data7'] = {}
    expected_output['data8'] = {}
    expected_output['data9'] = {'TEL':'Phone_Number', 'FAX':'Phone_Number', 'E_MAIL':'Email_Address',
                                'ADD1':'Street_Address','ADD2':'Street_Address','CEO_MD': 'Full_Name',
                                'G_MANAGER': 'Full_Name'}
    expected_output['data10'] = {'ADD1': 'Street_Address', 'ADD2': 'Street_Address', 'TEL': 'Phone_Number',
                                 'FAX': 'Phone_Number', 'E_MAIL': 'Email_Address', 'CEO_MD': 'Full_Name',
                                 'G_MANAGER': 'Full_Name'}

    expected_output['data11'] = {"ADD1":"Street_Address", "ADD2":"Street_Address","Telephone":"Phone_Number",
                                 "Fax":"Phone_Number","E-Mail Address":"Email_Address","CEO & Managing Director":"Full_Name",
                                 "General Manager":"Full_Name"}
    expected_output['data12'] = {"Address":"Full_Address","E-mail":"Email_Address"}
    expected_output['data13'] = {"gender":"Gender","email":"Email_Address","phone":"Phone_Number",
                                 "street_address":"Street_Address"}
    expected_output['data14'] = {"CODE_GENDER":"Gender"}
    expected_output['data15'] = {}
    expected_output['data16'] = {"credit card":"Credit_Card_Number"}
    expected_output['data17'] = {}
    expected_output['data18'] = {"name":"Full_Name","gender":"Gender"}
    expected_output['data19'] = {"name":"Full_Name","gender":"Gender"}
    expected_output['data20'] = {"sex":"Gender"}
    expected_output['data21'] = {"MetadataFrom":"Full_Name"}
    expected_output['data22'] = {"Name":"Full_Name"}
    expected_output['data23'] = {"sex":"Gender"}
    expected_output['data24'] = {"athlete_name":"Full_Name","athlete_sex":"Gender"}
    expected_output['data25'] = {"Name":"Full_Name"}
    expected_output['data26'] = {"name":"Full_Name","gender":"Gender"}
    for dataset in public_data_list.keys():
        for col in public_data_list[dataset].columns:
            if col not in expected_output[dataset].keys():
                expected_output[dataset][col] = "no_infotype"
            if expected_output[dataset][col] not in infotypes_to_use:
                expected_output[dataset][col] = "no_infotype"
    return expected_output


def get_best_infotype_pred(public_data_list,confidence_threshold ):
    column_info_list = populate_column_info_list(public_data_list)
    column_info_pred_list = check_predict_infotype(column_info_list, confidence_threshold, input_dict)
    public_data_predicted_infotype = dict()
    for dataset in public_data_list.keys():
        public_data_predicted_infotype[dataset] = {}
        for col in public_data_list[dataset].columns:
            for col_info in column_info_pred_list:
                if col_info.metadata.name == col and col_info.metadata.dataset_name == dataset:
                    public_data_predicted_infotype[dataset][col] = "no_infotype"
                    if len(col_info.infotype_proposals)>0:
                        highest_confidence_level = 0
                        for i in range(len(col_info.infotype_proposals)):
                            if col_info.infotype_proposals[i].confidence_level > highest_confidence_level:
                                public_data_predicted_infotype[dataset][col] = col_info.infotype_proposals[i].infotype
                                highest_confidence_level = col_info.infotype_proposals[i].confidence_level
    return public_data_predicted_infotype


def get_pred_exp_infotype_mapping(public_data_predicted_infotype,public_data_expected_infotype):
    mapping= []
    for dataset in public_data_predicted_infotype.keys():
        for col in public_data_predicted_infotype[dataset].keys():
            mapping.append((dataset, col, public_data_predicted_infotype[dataset][col] , public_data_expected_infotype[dataset][col]))
    return mapping

def get_prediction_statistics(mapping, infotypes_to_use,confidence_threshold):

    infotypes_to_use.append("no_infotype")
    all_infotypes = infotypes_to_use
    prediction_stats= pd.DataFrame()
    prediction_stats["InfoType_Name"] = all_infotypes
    prediction_stats["TP"] = 0
    prediction_stats["FP"] = 0
    prediction_stats["TN"] = 0
    prediction_stats["FN"] = 0
    for data_name, col_name, pred_val, true_val in mapping:
        if pred_val == true_val:
            current_tp_value = prediction_stats.loc[prediction_stats["InfoType_Name"] == true_val, "TP"].values
            current_tp_value += 1
            prediction_stats.loc[prediction_stats["InfoType_Name"] == true_val, "TP"] = current_tp_value
            true_neg_infotypes = [infotype for infotype in all_infotypes if infotype != true_val ]
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

    y_true = [s[3] for s in mapping]
    y_pred = [s[2] for s in mapping]
    df_confusion_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred, labels = all_infotypes ),
                                       columns = all_infotypes, index = all_infotypes)

    print("*************Prediction Statistics***************************")
    print(prediction_stats)
    print("********************")
    print(df_confusion_matrix)
    prediction_stats.to_csv(f".\Prediction_statistics_{confidence_threshold}.csv", index=False)
    df_confusion_matrix.to_csv(f".\confusion_matrix_{confidence_threshold}.csv")


public_data_list = get_public_data(input_data_dir)
public_data_predicted_infotype= get_best_infotype_pred(public_data_list,confidence_threshold)
public_data_expected_infotype = get_public_data_expected_output(public_data_list, infotypes_to_use)
infotype_mapping = get_pred_exp_infotype_mapping (public_data_predicted_infotype, public_data_expected_infotype)
get_prediction_statistics(infotype_mapping, infotypes_to_use,confidence_threshold)

@pytest.mark.parametrize("dataset_name,column_name, output_proposals, expected_output", [(a,b,c,d) for a,b,c,d in infotype_mapping])
def test_public_datasets(dataset_name, column_name, output_proposals, expected_output):
    assert output_proposals == expected_output, f"failed for column '{column_name}' in {dataset_name}"
