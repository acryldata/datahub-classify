import itertools
import os

import numpy as np
import pandas as pd

from datahub_classify.helper_classes import (
    ColumnInfo,
    ColumnMetadata,
    TableInfo,
    TableMetadata,
)
from datahub_classify.similarity_predictor import check_similarity, preprocess_tables

SEED = 42
use_embeddings = False
PRUNING_THRESHOLD = 0.8
platforms = ["A", "B", "C", "D", "E"]


def populate_tableinfo_object(df, dataset_name):
    """populate table info object for a dataset"""

    np.random.seed(SEED)
    table_meta_info = {
        "name": dataset_name,
        "description": f"This table contains description of {dataset_name}",
        "platform": platforms[np.random.randint(0, 5)],
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


current_wdr = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_wdr, "datasets")

df_1 = pd.read_csv(os.path.join(input_dir, "train.csv"))
df_2 = pd.read_csv(os.path.join(input_dir, "vehicles_1.csv"))
df_3 = pd.read_csv(os.path.join(input_dir, "vehicles_2.csv"))

table1_info = populate_tableinfo_object(df_1, "train")
table2_info = populate_tableinfo_object(df_2, "vehicles_1")
table3_info = populate_tableinfo_object(df_3, "vehicles_2")


if use_embeddings:
    table_info_list = preprocess_tables([table1_info, table2_info, table3_info])
else:
    table_info_list = [table1_info, table2_info, table3_info]

table_infos = {
    '"train"': table_info_list[0],
    "vehicles_1": table_info_list[1],
    "vehicles_2": table_info_list[2],
}
table_pairs = list(itertools.combinations(table_infos.keys(), 2))

print("Running Check_Similarity..................")
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
    post_pruning_mode_results[comb] = check_similarity(
        table_infos[comb[0]],
        table_infos[comb[1]],
        pruning_mode=False,
        use_embeddings=use_embeddings,
    )

print(pruning_mode_results)
print("============================================")
print(post_pruning_mode_results)
