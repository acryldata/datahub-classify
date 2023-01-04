from datahub_classify.helper_classes import (
    ColumnInfo,
    ColumnMetadata,
    TableInfo,
    TableMetadata,
)
from datahub_classify.similarity_predictor import check_similarity

table1_meta_info = {
    "Name": "data1",
    "Description": "This table contains description of data1",
    "Platform": "SQL",
    "Table_Id": "data1",
}

table2_meta_info = {
    "Name": "data2",
    "Description": "This table contains description of 2018-seattle-business-districts",
    "Platform": "PostgreSQL",
    "Table_Id": "data2",
}

table1_column_list = [
    ColumnInfo(
        metadata=ColumnMetadata(
            meta_info={
                "Name": "First and Last Name",
                "Description": " First and Last Name",
                "Datatype": "str",
                "Dataset_Name": "data1",
                "Column_Id": "table1_col1",
            }
        ),
        parent_columns=[],
        values=[],
        infotype_proposals=None,
    ),
    ColumnInfo(
        metadata=ColumnMetadata(
            meta_info={
                "Name": "Age",
                "Description": "Age of the candidate",
                "Datatype": "str",
                "Dataset_Name": "data1",
                "Column_Id": "table1_col2",
            }
        ),
        parent_columns=[],
        values=[],
        infotype_proposals=None,
    ),
]

table2_column_list = [
    ColumnInfo(
        metadata=ColumnMetadata(
            meta_info={
                "Name": "Name",
                "Description": "Employee Name",
                "Datatype": "str",
                "Dataset_Name": "data2",
                "Column_Id": "table2_col1",
            }
        ),
        parent_columns=[],
        values=[],
        infotype_proposals=None,
    ),
    ColumnInfo(
        metadata=ColumnMetadata(
            meta_info={
                "Name": "Person's Age",
                "Description": " Organization",
                "Datatype": "int64",
                "Dataset_Name": "data2",
                "Column_Id": "table2_col2",
            }
        ),
        parent_columns=[],
        values=[],
        infotype_proposals=None,
    ),
]

table_1_info = TableInfo(
    metadata=TableMetadata(meta_info=table1_meta_info),
    column_infos=table1_column_list,
    parent_tables=[],
)

table_2_info = TableInfo(
    metadata=TableMetadata(meta_info=table2_meta_info),
    column_infos=table2_column_list,
    parent_tables=[],
)


def main():
    print("Checking for Similarity......")

    try:
        overall_table_similarity_score, column_similarity_scores = check_similarity(
            table_1_info, table_2_info
        )
        print(f"Overall Table Similarity Score: {overall_table_similarity_score}")
        print(f"Column Similarity Scores: {column_similarity_scores}")
    except Exception as e:
        print(f"Error in calculation due to {e}")


if __name__ == "__main__":
    main()
