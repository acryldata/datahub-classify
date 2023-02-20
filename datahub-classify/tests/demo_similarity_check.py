import itertools
import os
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from datahub_classify.helper_classes import (
    ColumnInfo,
    ColumnMetadata,
    TableInfo,
    TableMetadata,
)
from datahub_classify.similarity_predictor import check_similarity, preprocess_tables

# import time
# from typing import List

print("libraries Imported..................")
SEED = 42
use_embeddings = False
# np.random.seed(SEED)
PRUNING_THRESHOLD = 0.8
platforms = ["A", "B", "C", "D", "E"]


def populate_tableinfo_object(df, dataset_name):
    """populate table info object for a dataset"""

    np.random.seed(SEED)
    table_meta_info = {
        "Name": dataset_name,
        "Description": f"This table contains description of {dataset_name}",
        "Platform": platforms[np.random.randint(0, 5)],
        "Table_Id": dataset_name,
    }
    col_infos = []
    for col in df.columns:
        fields = {
            "Name": col,
            "Description": f" {col}",
            "Datatype": str(df[col].dropna().dtype),
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


def populate_similar_tableinfo_object(df, dataset_name, random_df):
    """populate table info object for a dataset by randomly adding some additional
    columns to the dataset, thus obtain a logical copy of input dataset"""
    # df = public_data_list[dataset_name].copy()
    # random_df_key = list(public_data_list.keys())[
    #     np.random.randint(0, len(public_data_list))
    # ]
    # while random_df_key == dataset_name:
    #     random_df_key = list(public_data_list.keys())[
    #         np.random.randint(0, len(public_data_list))
    #     ]
    # random_df = public_data_list[random_df_key].copy()
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
    # fmt:off
    common_variations = ["#", "$", "%", "&", "*", "-", ".", ":", ";", "?",
                         "@", "_", "~", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                         ]
    # fmt:on
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


model = SentenceTransformer("all-MiniLM-L6-v2")

input_dir = "C:/PROJECTS/Acryl/acryl_git/datahub-classify/tests/demo_test_csv"
df_1 = pd.read_csv(os.path.join(input_dir, "Customer_data.csv"))
print(df_1.shape)
df_2 = pd.read_csv(os.path.join(input_dir, "vehicles_1.csv"))
print(df_2.shape)
# df_3 = pd.read_csv(os.path.join(input_dir, "Cust_data.csv"))
# print(df_3.shape)
df_3 = pd.read_csv(os.path.join(input_dir, "vehicles_2.csv"))
print(df_3.shape)
# df_3 = pd.read_csv(os.path.join(input_dir, "vehicles_1.csv"))

table1_info = populate_tableinfo_object(df_1, "Customer_data")
table2_info = populate_tableinfo_object(df_2, "vehicles_1")
table3_info = populate_tableinfo_object(df_3, "vehicles_2")
# table4_info = populate_tableinfo_object(df_4, "vehicles_1")
# table2_info = populate_tableinfo_object(df_2, "vehicles_2.csv")
# table2_info = populate_similar_tableinfo_object(df_1, "Book1.xlsx-credit-card-number", df_2)

if use_embeddings:
    table_info_list = preprocess_tables([table1_info, table2_info, table3_info])
else:
    table_info_list = [table1_info, table2_info, table3_info]

table_infos = {
    '"Customer_data"': table_info_list[0],
    "vehicles_1": table_info_list[1],
    "vehicles_2": table_info_list[2],
}
table_pairs = list(itertools.combinations(table_infos.keys(), 2))

print("Running Check_Similarity..................")
# begin = time.time()
pruning_mode_results = {}

for pair in table_pairs:
    pruning_mode_results[pair] = check_similarity(
        table_infos[pair[0]],
        table_infos[pair[1]],
        pruning_mode=True,
        use_embeddings=use_embeddings,
    )

post_pruning_mode_pairs = [
    key
    for key, value in pruning_mode_results.items()
    if value[0].score > PRUNING_THRESHOLD
]

post_pruning_mode_results = {}
for comb in post_pruning_mode_pairs:
    # tables = comb.split("_SPLITTER_")
    post_pruning_mode_results[comb] = check_similarity(
        table_infos[comb[0]],
        table_infos[comb[1]],
        pruning_mode=False,
        use_embeddings=use_embeddings,
    )

# end = time.time()
# print(df_1.info())
# print(df_2.info())
# print(df_3.info())
print(pruning_mode_results)
# print(post_pruning_mode_pairs)
print(post_pruning_mode_results)


# for column_pair in column_similarity_scores.keys():
#     print(column_pair)
#     print(column_similarity_scores[column_pair])
# print(f"Time taken {round((end - begin), 10)} seconds")
# print(column_similarity_scores)
