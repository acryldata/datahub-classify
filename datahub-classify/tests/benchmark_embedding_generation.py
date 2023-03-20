import copy
import itertools
import logging
import os
import pickle
import time
from typing import List

from datahub_classify.helper_classes import TableInfo
from datahub_classify.similarity_predictor import preprocess_tables

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

NUM_COPIES = 1

CURRENT_WDR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(CURRENT_WDR, "test_input")
TABLE_INFOS_PATH = os.path.join(INPUT_DIR, "table_info_objects.pickle")
TABLE_INFO_COPIES_PATH = os.path.join(INPUT_DIR, "logical_copies.pickle")


def get_table_info_objects(copies_count: int) -> List[TableInfo]:
    with open(TABLE_INFOS_PATH, "rb") as table_info_file:
        table_infos_ = pickle.load(table_info_file)
    with open(TABLE_INFO_COPIES_PATH, "rb") as table_info_copies_file:
        table_info_copies = pickle.load(table_info_copies_file)
    if table_info_copies:
        table_infos_.update(table_info_copies)
    table_infos_ = list(table_infos_.values())

    table_info_copies = []
    if copies_count > 0:
        for obj in table_infos_:
            table_info_copies.extend(
                list(itertools.repeat(copy.deepcopy(obj), copies_count))
            )
    table_infos_.extend(table_info_copies)
    return table_infos_


logger.info("Preparing Tables Info Objects.............")
table_infos = get_table_info_objects(NUM_COPIES)

logger.info(f"Total Number of Tables: {len(table_infos)}")
embedding_generation_start_time = time.time()
output = preprocess_tables(table_infos)
embedding_generation_end_time = time.time()

with open("Embedding_Generation_Timings.txt", "a") as f:
    f.write(
        f"Embedding Generation Time for {len(table_infos)} tables is - {embedding_generation_end_time - embedding_generation_start_time} sec\n\n"
    )
print(
    f"Embedding Generation Time for {len(table_infos)} tables is - {embedding_generation_end_time - embedding_generation_start_time} sec\n\n"
)
