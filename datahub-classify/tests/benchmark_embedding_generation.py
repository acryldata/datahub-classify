import glob
import logging
import os
import re
import time

import numpy as np
import pandas as pd

from datahub_classify.helper_classes import (
    ColumnInfo,
    ColumnMetadata,
    TableInfo,
    TableMetadata,
)
from datahub_classify.similarity_predictor import preprocess_tables

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.info("libraries Imported..................")


NUM_COPIES = 1
SEED = 100
np.random.seed(SEED)
PLATFORMS = ["A", "B", "C", "D", "E"]


def load_df(dataset_name):
    path = all_datasets_paths[dataset_name]
    if path.endswith("csv"):
        df = pd.read_csv(path, nrows=2)
    elif path.endswith("xlsx"):
        df = pd.read_excel(path, nrows=2)
    else:
        df = None
    return df


def populate_tableinfo_object(dataset_name):
    """populate table info object for a dataset"""
    df = load_df(dataset_name)
    np.random.seed(SEED)
    table_meta_info = {
        "name": dataset_name,
        "description": f"This table contains description of {dataset_name}",
        "platform": PLATFORMS[np.random.randint(0, 5)],
        "table_id": dataset_name,
    }
    col_infos = []
    for col in df.columns:
        fields = {
            "name": col,
            "description": f" {col}",
            "datatype": str(df[col].dropna().dtype),
            "dataset_name": dataset_name,
            "column_id": dataset_name + "_SPLITTER_" + col,
        }
        metadata_col = ColumnMetadata(**fields)
        # parent_cols = list()
        col_info_ = ColumnInfo(metadata_col)
        col_infos.append(col_info_)

    metadata_table = TableMetadata(**table_meta_info)
    # parent_tables = list()
    table_info = TableInfo(metadata_table, col_infos)
    return table_info


def populate_similar_tableinfo_object(dataset_name):
    """populate table info object for a dataset by randomly adding some additional
    columns to the dataset, thus obtain a logical copy of input dataset"""
    df = load_df(dataset_name)
    np.random.seed(SEED)
    random_df_key = list(
        key for key in all_datasets_paths.keys() if key != dataset_name
    )[np.random.randint(0, len(all_datasets_paths) - 1)]
    random_df = load_df(random_df_key).copy()
    random_df_columns = [col for col in random_df.columns if col not in df.columns]
    random_df = random_df[random_df_columns]
    random_df.columns = ["data2" + "_" + col for col in random_df.columns]
    df.columns = ["data1" + "_" + col for col in df.columns]
    second_df = pd.concat([df, random_df], axis=1)
    if len(random_df_columns) < 2:
        cols_to_keep = list(df.columns) + list(random_df.columns)
    else:
        cols_to_keep = list(df.columns) + list(random_df.columns[:2])
    second_df = second_df[cols_to_keep]
    np.random.seed(SEED)
    table_meta_info = {
        "name": dataset_name + "_LOGICAL_COPY",
        "description": f" {dataset_name}",
        "platform": PLATFORMS[np.random.randint(0, 5)],
        "table_id": dataset_name + "_LOGICAL_COPY",
    }
    col_infos = []
    swap_case = ["yes", "no"]
    # fmt:off
    common_variations = ["#", "$", "%", "&", "*", "-", ".", ":", ";", "?",
                         "@", "_", "~", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                         ]
    # fmt:on
    np.random.seed(SEED)
    for col in second_df.columns:
        col_name = col.split("_", 1)[1]
        col_name_with_variation = str(
            common_variations[np.random.randint(0, len(common_variations))]
        )
        np.random.seed(SEED)
        for word in re.split("[^A-Za-z]", col_name):
            random_variation = str(
                common_variations[np.random.randint(0, len(common_variations))]
            )
            np.random.seed(SEED)
            is_swap_case = swap_case[np.random.randint(0, 2)]
            if is_swap_case:
                word = word.swapcase()
            col_name_with_variation = col_name_with_variation + word + random_variation

        fields = {
            "name": col_name_with_variation,
            "description": f'{col.split("_", 1)[1]}',
            "datatype": str(second_df[col].dropna().dtype),
            "dataset_name": dataset_name + "_LOGICAL_COPY",
            "column_id": dataset_name
            + "_LOGICAL_COPY_"
            + "_SPLITTER_"
            + col.split("_", 1)[1],
        }
        metadata_col = ColumnMetadata(**fields)
        parent_cols = [col if col in df.columns else None]
        col_info_ = ColumnInfo(metadata_col)
        col_info_.parent_columns = parent_cols
        col_infos.append(col_info_)
    metadata_table = TableMetadata(**table_meta_info)
    parent_tables = [dataset_name]
    table_info = TableInfo(metadata_table, col_infos, parent_tables)
    return table_info


current_wdr = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_wdr, "datasets")

all_datasets_paths = {
    os.path.basename(file1).rsplit(".", 1)[0]: file1
    for file1 in glob.glob(f"{input_dir}/*")
}

logger.info("Creating Tables Info Objects.............")
table_info_list = [populate_tableinfo_object(key) for key in all_datasets_paths.keys()]

for key in all_datasets_paths.keys():
    for i in range(NUM_COPIES):
        table_info_list.append(populate_similar_tableinfo_object(key))

logger.info(f"Total Number of Tables: {len(table_info_list)}")
embedding_generation_start_time = time.time()
output = preprocess_tables(table_info_list)
embedding_generation_end_time = time.time()

with open("Embedding_Generation_Timings.txt", "a") as f:
    f.write(
        f"Embedding Generation Time for {len(table_info_list)} tables is - {embedding_generation_end_time - embedding_generation_start_time} sec\n\n"
    )
print(
    f"Embedding Generation Time for {len(table_info_list)} tables is - {embedding_generation_end_time - embedding_generation_start_time} sec\n\n"
)
