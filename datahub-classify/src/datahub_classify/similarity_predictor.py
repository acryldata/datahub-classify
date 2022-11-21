
from helper_classes import TableInfo
from utils import compute_table_similarity, compute_column_similarity, read_glove_vector



def check_similarity(table_info1: TableInfo, table_info2: TableInfo):
    print("finding table similarity")
    overall_table_similarity_score = compute_table_similarity(table_info1, table_info2)
    column_similarity_scores = {}
    print("====================================")
    print("********************finding column similarities**********************")
    print("total pairs --> ", len(table_info1.column_infos)* len(table_info2.column_infos))
    for col_info1 in table_info1.column_infos:
        for col_info2 in table_info2.column_infos:
            # print("processing pair: ", (col_info1.metadata.name, col_info2.metadata.name))
            overall_column_similarity_score = compute_column_similarity(col_info1, col_info2, overall_table_similarity_score)
            column1_id = col_info1.metadata.column_id
            column2_id = col_info2.metadata.column_id
            column_similarity_scores[(column1_id, column2_id)] = overall_column_similarity_score
    return overall_table_similarity_score, column_similarity_scores
