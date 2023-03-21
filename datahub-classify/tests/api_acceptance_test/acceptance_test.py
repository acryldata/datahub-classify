import glob
import itertools
import logging
import os
from typing import Dict, List, Tuple
import pandas as pd

from datahub_classify.helper_classes import (
    ColumnInfo,
    ColumnMetadata,
    TableInfo,
    TableMetadata,
)
from datahub_classify.similarity_predictor import check_similarity, preprocess_tables

logger = logging.getLogger(__name__)

PRUNING_THRESHOLD = 0.8
FINAL_THRESHOLD = 0.6
COLUMN_SIMILARITY_THRESHOLD = 0.8
CURRENT_WDR = os.path.dirname(os.path.abspath(__file__))
USE_EMBEDDINGS = True
INPUT_DIR = f"{CURRENT_WDR}/datasets"
all_datasets_paths = {
    os.path.basename(file).rsplit(".", 1)[0]: file
    for file in glob.glob(f"{INPUT_DIR}/*")
}


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
    # np.random.seed(SEED)
    table_meta_info = {
        "name": dataset_name,
        "description": f"This table contains description of {dataset_name}",
        "platform": "A",
        # PLATFORMS[np.random.randint(0, 5)],
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


def get_table_infos_and_pairs() -> Tuple[Dict[str, TableInfo], List[Tuple[str, str]]]:
    table_infos_ = {
        key: populate_tableinfo_object(key) for key in all_datasets_paths.keys()
    }
    table_pairs_ = list(itertools.combinations(table_infos_.keys(), 2))

    return table_infos_, table_pairs_


def get_similarity_predictions(
    table_infos_: Dict[str, TableInfo], table_pairs_: List[Tuple[str, str]]
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, float]]:
    columns_predicted_scores_: Dict[str, float] = dict()
    pruning_mode_results: Dict[str, Tuple] = dict()
    non_pruning_mode_results: Dict[str, Tuple] = dict()

    logger.info("Starting check similarity in pruning mode.............")
    for table_pair in table_pairs_:
        table_pair_list = sorted(table_pair, key=str.lower)
        table_pair = (table_pair_list[0], table_pair_list[1])
        pruning_mode_results[
            f"{table_pair[0]}_SPLITTER_{table_pair[1]}"
        ] = check_similarity(
            table_infos_[table_pair[0]],
            table_infos_[table_pair[1]],
            pruning_mode=True,
            use_embeddings=False,
        )
    pruning_mode_output_predicted_ = {
        key: ("not_similar" if value[0].score < PRUNING_THRESHOLD else "similar")
        for key, value in pruning_mode_results.items()
    }

    non_pruning_mode_combinations = [
        key
        for key, value in pruning_mode_output_predicted_.items()
        if value == "similar"
    ]

    non_pruning_table_infos = None
    non_pruning_table_keys = []
    # non_pruning_table_infos_list = []
    if USE_EMBEDDINGS:
        for pair in non_pruning_mode_combinations:
            tables = pair.split("_SPLITTER_")
            if tables[0] not in non_pruning_table_keys:
                non_pruning_table_keys.append(tables[0])
            if tables[1] not in non_pruning_table_keys:
                non_pruning_table_keys.append(tables[1])

        non_pruning_table_keys = sorted(non_pruning_table_keys)
        non_pruning_table_infos_list = [table_infos_[key] for key in non_pruning_table_keys]

        non_pruning_table_infos_list = preprocess_tables(non_pruning_table_infos_list)

        non_pruning_table_infos = {
            key: value
            for key, value in zip(non_pruning_table_keys, non_pruning_table_infos_list)
        }

    if USE_EMBEDDINGS and non_pruning_table_infos:
        table_infos_ = non_pruning_table_infos

    logger.info("Starting check similarity in non pruning mode.............")
    for comb in non_pruning_mode_combinations:
        tables = comb.split("_SPLITTER_")
        non_pruning_mode_results[comb] = check_similarity(
            table_infos_[tables[0]],
            table_infos_[tables[1]],
            pruning_mode=False,
            use_embeddings=USE_EMBEDDINGS,
        )

    non_pruning_mode_output_predicted_ = {
        key: ("not_similar" if value[0].score < FINAL_THRESHOLD else "similar")
        for key, value in non_pruning_mode_results.items()
    }
    for i, data_pair in enumerate(non_pruning_mode_results.keys()):
        for key, value in non_pruning_mode_results[data_pair][1].items():
            columns_predicted_scores_[key] = value.score
    with open(os.path.join(CURRENT_WDR, "acceptance_pruning_results.json"), "w") as f:
        print(pruning_mode_results, file=f)
    with open(
        os.path.join(CURRENT_WDR, "acceptance_nonpruning_results.json"), "a"
    ) as f:
        print(non_pruning_mode_results, file=f)

    return (
        pruning_mode_output_predicted_,
        non_pruning_mode_output_predicted_,
        columns_predicted_scores_,
    )


table_infos, table_pairs = get_table_infos_and_pairs()
(
    pruning_mode_output_PREDICTED,
    non_pruning_mode_output_PREDICTED,
    columns_predicted_scores,
) = get_similarity_predictions(table_infos_=table_infos, table_pairs_=table_pairs)
