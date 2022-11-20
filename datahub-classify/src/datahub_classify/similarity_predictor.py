
from helper_classes import TableInfo
from infotype_utils import compute_table_similarity, compute_column_similarity

def check_similarity(table_info1: TableInfo, table_info2: TableInfo):
    overall_table_similarity_score = compute_table_similarity(table_info1, table_info2)
    column_similarity_scores = {}
    for col_info1 in table_info1.column_infos:
        for col_info2 in table_info2.column_infos:
            overall_column_similarity_score = compute_column_similarity(col_info1, col_info2, overall_table_similarity_score)
            column1_id = col_info1.metadata.column_id
            column2_id = col_info2.metadata.column_id
            column_similarity_scores[(column1_id, column2_id)] = overall_column_similarity_score
    return overall_table_similarity_score, column_similarity_scores
