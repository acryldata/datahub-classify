import logging
from typing import List

from sentence_transformers import SentenceTransformer

from datahub_classify.helper_classes import TableInfo, TextEmbeddings

logger = logging.getLogger(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(table_info_list: List[TableInfo]) -> List[TableInfo]:
    logger.info("** Generating Embeddings **")
    try:
        all_strings = []
        for table_info in table_info_list:
            logger.info(
                f"** Generating Embeddings for {table_info.metadata.table_id} **"
            )
            if table_info.metadata.name:
                all_strings.append(table_info.metadata.name.lower().strip())
            if table_info.metadata.description:
                all_strings.append(table_info.metadata.description.lower().strip())
            for col_info in table_info.column_infos:
                if col_info.metadata.name:
                    all_strings.append(col_info.metadata.name.lower().strip())
                if col_info.metadata.description:
                    all_strings.append(col_info.metadata.description.lower().strip())
        all_embeddings = model.encode(all_strings)
        all_strings_with_embeddings = {
            key: value for key, value in zip(all_strings, all_embeddings)
        }

        for table_info in table_info_list:
            if table_info.metadata.name:
                table_info.metadata.name_embedding.append(
                    TextEmbeddings(
                        "sentence_transformer",
                        all_strings_with_embeddings[
                            table_info.metadata.name.lower().strip()
                        ],
                    )
                )
            if table_info.metadata.description:
                table_info.metadata.desc_embedding.append(
                    TextEmbeddings(
                        "sentence_transformer",
                        all_strings_with_embeddings[
                            table_info.metadata.description.lower().strip()
                        ],
                    )
                )
            for col_info in table_info.column_infos:
                if col_info.metadata.name:
                    col_info.metadata.name_embedding.append(
                        TextEmbeddings(
                            "sentence_transformer",
                            all_strings_with_embeddings[
                                col_info.metadata.name.lower().strip()
                            ],
                        )
                    )
                if col_info.metadata.description:
                    col_info.metadata.desc_embedding.append(
                        TextEmbeddings(
                            "sentence_transformer",
                            all_strings_with_embeddings[
                                col_info.metadata.description.lower().strip()
                            ],
                        )
                    )
    except Exception as e:
        logger.error("Failed to Generate Embeddings", exc_info=e)
    return table_info_list
