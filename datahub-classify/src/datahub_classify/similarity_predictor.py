
from helper_classes import TableInfo
from utils import compute_table_similarity, compute_column_similarity


def check_similarity(table_info1: TableInfo, table_info2: TableInfo):
    print(f"Finding table similarity between Table {table_info1.metadata.table_id} and {table_info2.metadata.table_id}")
    overall_table_similarity_score = None
    try:
        overall_table_similarity_score = compute_table_similarity(table_info1, table_info2)
    except Exception as e:
        print(f"Failed to compute table similarity between Table {table_info1.metadata.table_id} and {table_info2.metadata.table_id}")
        print(f"Error - {str(e)}")

    column_similarity_scores = {}
    print("====================================")
    print("******************** Finding column similarities **********************")
    print(f"Total pairs --> {len(table_info1.column_infos)* len(table_info2.column_infos)}")
    # TODO: if we find a column pair with high similarity score, can we skip the pair in further processing
    # TODO: hypothesis is if table1.colA matches with table2.colB then we will not have any other column in table2 matching with table1.colA
    for col_info1 in table_info1.column_infos:
        for col_info2 in table_info2.column_infos:
            print(f"Processing pair: {(col_info1.metadata.name, col_info2.metadata.name)}")
            overall_column_similarity_score = None
            try:
                overall_column_similarity_score = compute_column_similarity(col_info1, col_info2, overall_table_similarity_score)
            except Exception as e:
                print(f"Failed to compute column similarity between Column {col_info1.metadata.column_id} and {col_info1.metadata.column_id}")
                print(f"Error - {str(e)}")
            column1_id = col_info1.metadata.column_id
            column2_id = col_info2.metadata.column_id
            column_similarity_scores[(column1_id, column2_id)] = overall_column_similarity_score
    return overall_table_similarity_score, column_similarity_scores
