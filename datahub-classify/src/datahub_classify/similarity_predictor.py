import logging
import os
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

import datahub_classify.similarity_numeric_constants as config
from datahub_classify.helper_classes import (
    ColumnInfo,
    DebugInfo,
    SimilarityInfo,
    TableInfo,
    TextEmbeddings
)
from datahub_classify.utils import (
    compute_string_similarity,
    load_stopwords,
    read_glove_vector,
)

logger = logging.getLogger(__name__)
current_wdr = os.path.dirname(os.path.abspath(__file__))
glove_vec = os.path.join(current_wdr, "glove.6B.50d.txt")
model = SentenceTransformer("all-MiniLM-L6-v2")
stop_words = load_stopwords()
use_embeddings = True

if not os.path.isfile(glove_vec):
    from datahub_classify.utils import download_glove_embeddings

    logger.debug("Downloading GLOVE Embeddings.............")
    download_glove_embeddings(glove_vec)

word_to_vec_map = read_glove_vector(glove_vec)


def column_dtype_similarity(column_1_dtype: str, column_2_dtype: str) -> Optional[int]:
    try:
        if column_1_dtype == column_2_dtype:
            column_dtype_score = 1
        else:
            column_dtype_score = 0
    except Exception as e:
        logger.error(
            f"Failed to compute column dtype similarity for '{column_1_dtype}' and '{column_2_dtype}' ",
            exc_info=e,
        )
        column_dtype_score = None
    return column_dtype_score


def table_platform_similarity(
        platform_1_name: str, platform_2_name: str
) -> Optional[int]:
    try:
        if platform_1_name != platform_2_name:
            platform_score = 1
        else:
            platform_score = 0
    except Exception as e:
        logger.error("Failed to compute platform score", exc_info=e)
        platform_score = None
    return platform_score


def compute_lineage_score(
        entity_1_parents: List, entity_2_parents: List, entity_1_id: str, entity_2_id: str
) -> Optional[int]:
    try:
        if (entity_1_id in entity_2_parents) or (entity_2_id in entity_1_parents):
            lineage_score = 1
        else:
            lineage_score = 0
    except Exception as e:
        logger.error(
            f"Failed to compute lineage score for entities with IDs: '{entity_1_id}' and '{entity_2_id}'",
            exc_info=e,
        )
        lineage_score = None
    return lineage_score


def table_schema_similarity(
        col_infos1: List[ColumnInfo],
        col_infos2: List[ColumnInfo],
        word_vec_map: dict,
        pair_score_threshold: float = config.schema_col_pair_score_threshold,
) -> Optional[float]:
    try:
        matched_pairs = []
        num_cols_table1 = len(col_infos1)
        num_cols_table2 = len(col_infos2)
        columns_to_remove = []
        for col1 in col_infos1:
            for col2 in col_infos2:
                if col2.metadata.column_id in columns_to_remove:
                    continue
                col_name_score = compute_string_similarity(
                    col1.metadata.name,
                    col2.metadata.name,
                    col1.metadata.name_embedding,
                    col2.metadata.name_embedding,
                    text_type="name",
                    word_to_vec_map=word_vec_map,
                    use_embeddings=use_embeddings,
                    stop_words=stop_words,
                )
                col_dtype_score = column_dtype_similarity(
                    col1.metadata.datatype, col2.metadata.datatype
                )
                if col_name_score is not None and col_dtype_score is not None:
                    pair_score = (
                            config.schema_col_name_weight * col_name_score
                            + config.schema_col_dtype_weight * col_dtype_score
                    )

                    if pair_score > pair_score_threshold:
                        matched_pairs.append(
                            (col1.metadata.column_id, col2.metadata.column_id)
                        )
                        # TODO: table_2_cols_names_dtypes dict is getting used in the for loop and modified at runtime
                        # TODO: verify: it will not create any problem
                        columns_to_remove.append(col2.metadata.column_id)
                        break
        schema_score = 2 * len(matched_pairs) / (num_cols_table1 + num_cols_table2)
    except Exception as e:
        logger.error("Failed to compute table schema similarity ", exc_info=e)
        schema_score = None
    return schema_score


def compute_table_overall_similarity_score(
        name_score: float,
        desc_score: Optional[float],
        platform_score: Optional[int],
        lineage_score: Optional[int],
        schema_score: float,
        # desc_present: bool,
) -> float:
    if platform_score is None:
        weighted_score = (
                config.overall_name_score_weight * name_score
                + config.overall_schema_score_weight * schema_score
        )
    else:
        weighted_score = (
                config.overall_name_score_weight * name_score
                + config.overall_platform_score_weight * platform_score
                + config.overall_schema_score_weight * schema_score
        )
    # TODO: can we think of something like, if platform_score is 1 then boost by x% and if 0 then boost by y%
    # TODO: here x < y, i.e. if platform_score 0 means less probability of having two tables as similar
    if desc_score is not None:
        if desc_score >= config.table_desc_threshold:
            weighted_score = np.minimum(config.overall_desc_boost * weighted_score, 1)
        if desc_score < config.overall_desc_score_threshold:
            weighted_score = config.overall_desc_penalty * weighted_score
    overall_table_similarity_score = weighted_score
    if (
            weighted_score >= config.table_weighted_score_threshold
            and lineage_score is not None
            and lineage_score == 1
    ):
        overall_table_similarity_score = 1
    return np.round(overall_table_similarity_score, 2)


def compute_column_overall_similarity_score(
        name_score: float,
        dtype_score: int,
        desc_score: Optional[float],
        table_similarity_score: Optional[float],
        lineage_score: Optional[int],
        # desc_present: bool,
) -> float:
    weighted_score = name_score
    if dtype_score == 1:
        weighted_score = config.column_dtype_boost * weighted_score
    else:
        weighted_score = config.column_dtype_penalty * weighted_score

    if desc_score is not None:
        if desc_score >= config.column_desc_threshold:
            weighted_score = config.column_desc_boost * weighted_score
        if desc_score < 0.5:
            weighted_score = config.column_desc_penalty * weighted_score

    if weighted_score > config.column_weighted_score_threshold:
        if (
                table_similarity_score is not None
                and table_similarity_score > config.table_similarity_threshold
        ):
            weighted_score = config.column_table_similarity_boost * weighted_score
        if lineage_score is not None and lineage_score == 1:
            weighted_score = config.column_lineage_boost * weighted_score

    overall_column_similarity_score = np.minimum(weighted_score, 1)
    return np.round(overall_column_similarity_score, 2)


def compute_table_similarity(
        table_info1: TableInfo, table_info2: TableInfo, word_vec_map: dict
) -> Tuple[Optional[float], DebugInfo]:
    # table_1_cols_name_dtypes = list()
    # for col_info in table_info1.column_infos:
    #     table_1_cols_name_dtypes.append(
    #         (col_info.metadata.name, col_info.metadata.datatype)
    #     )
    #
    # table_2_cols_name_dtypes = list()
    # for col_info in table_info2.column_infos:
    #     table_2_cols_name_dtypes.append(
    #         (col_info.metadata.name, col_info.metadata.datatype)
    #     )

    if table_info1.metadata.description and table_info2.metadata.description:
        desc_present = True
    else:
        desc_present = False

    table_name_score = compute_string_similarity(
        table_info1.metadata.name,
        table_info2.metadata.name,
        table_info1.metadata.name_embedding,
        table_info2.metadata.name_embedding,
        text_type="name",
        word_to_vec_map=word_vec_map,
        use_embeddings=use_embeddings,
        stop_words=stop_words,
    )
    if desc_present:
        table_desc_score = compute_string_similarity(
            table_info1.metadata.description,
            table_info2.metadata.description,
            table_info1.metadata.desc_embedding,
            table_info2.metadata.desc_embedding,
            text_type="desc",
            word_to_vec_map=word_vec_map,
            use_embeddings=use_embeddings,
            stop_words=stop_words,
        )
    else:
        table_desc_score = None
    table_platform_score = table_platform_similarity(
        table_info1.metadata.platform, table_info2.metadata.platform
    )
    table_lineage_score = compute_lineage_score(
        table_info1.parent_tables,
        table_info2.parent_tables,
        table_info1.metadata.table_id,
        table_info2.metadata.table_id,
    )
    table_schema_score = table_schema_similarity(
        table_info1.column_infos,
        table_info2.column_infos,
        word_vec_map=word_vec_map,
    )

    table_prediction_factor_confidence = DebugInfo(
        name=table_name_score,
        description=table_desc_score,
        platform=table_platform_score,
        lineage=table_lineage_score,
        table_schema=table_schema_score,
    )

    if table_name_score is None or table_schema_score is None:
        overall_table_similarity_score = None
    else:
        overall_table_similarity_score = compute_table_overall_similarity_score(
            table_name_score,
            table_desc_score,
            table_platform_score,
            table_lineage_score,
            table_schema_score,
        )
    return overall_table_similarity_score, table_prediction_factor_confidence


def compute_column_similarity(
        col_info1: ColumnInfo,
        col_info2: ColumnInfo,
        overall_table_similarity_score: Optional[float],
        word_vec_map: dict,
) -> Tuple[Optional[float], DebugInfo]:
    if col_info1.metadata.description and col_info2.metadata.description:
        desc_present = True
    else:
        desc_present = False

    column_name_score = compute_string_similarity(
        col_info1.metadata.name,
        col_info2.metadata.name,
        col_info1.metadata.name_embedding,
        col_info2.metadata.name_embedding,
        text_type="name",
        word_to_vec_map=word_vec_map,
        use_embeddings=use_embeddings,
        stop_words=stop_words,
    )
    if desc_present:
        column_desc_score = compute_string_similarity(
            col_info1.metadata.description,
            col_info2.metadata.description,
            col_info1.metadata.desc_embedding,
            col_info2.metadata.desc_embedding,
            text_type="desc",
            word_to_vec_map=word_vec_map,
            use_embeddings=use_embeddings,
            stop_words=stop_words,
        )
    else:
        column_desc_score = None
    column_dtype_score = column_dtype_similarity(
        col_info1.metadata.datatype, col_info2.metadata.datatype
    )
    column_lineage_score = compute_lineage_score(
        col_info1.parent_columns,
        col_info2.parent_columns,
        col_info1.metadata.column_id,
        col_info2.metadata.column_id,
    )

    col_prediction_factor_confidence = DebugInfo(
        name=column_name_score,
        description=column_desc_score,
        datatype=column_dtype_score,
        lineage=column_lineage_score,
        table_schema=overall_table_similarity_score,
    )

    if column_name_score is None or column_dtype_score is None:
        overall_column_similarity_score = None
    else:
        overall_column_similarity_score = compute_column_overall_similarity_score(
            column_name_score,
            column_dtype_score,
            column_desc_score,
            overall_table_similarity_score,
            column_lineage_score,
            # desc_present,
        )

    return overall_column_similarity_score, col_prediction_factor_confidence


def check_similarity(table_info1: TableInfo, table_info2: TableInfo) -> tuple:
    logger.info(
        f"** Finding table similarity between Table '{table_info1.metadata.table_id}' and '{table_info2.metadata.table_id}' **"
    )
    try:
        (
            overall_table_similarity_score,
            table_prediction_factor_confidence,
        ) = compute_table_similarity(table_info1, table_info2, word_to_vec_map)
    except Exception as e:
        logger.error(
            f"Failed to compute table similarity between Table "
            f"{table_info1.metadata.table_id} and {table_info2.metadata.table_id}",
            exc_info=e,
        )
        overall_table_similarity_score = None
        table_prediction_factor_confidence = None

    table_similarity_score = SimilarityInfo(
        score=overall_table_similarity_score,
        prediction_factor_confidence=table_prediction_factor_confidence,
    )

    column_similarity_scores = {}
    logger.info("** Finding column similarities **")
    logger.debug(
        f"Total pairs --> {len(table_info1.column_infos) * len(table_info2.column_infos)}"
    )
    for col_info1 in table_info1.column_infos:
        for col_info2 in table_info2.column_infos:
            logger.debug(
                f"Processing pair: {(col_info1.metadata.column_id, col_info2.metadata.column_id)}"
            )
            try:
                (
                    overall_column_similarity_score,
                    col_prediction_factor_confidence,
                ) = compute_column_similarity(
                    col_info1,
                    col_info2,
                    overall_table_similarity_score,
                    word_to_vec_map,
                )
            except Exception as e:
                logger.error(
                    f"Failed to compute column similarity between Column "
                    f"{col_info1.metadata.column_id} and {col_info1.metadata.column_id}",
                    exc_info=e,
                )
                overall_column_similarity_score = None
                col_prediction_factor_confidence = None

            column1_id = col_info1.metadata.column_id
            column2_id = col_info2.metadata.column_id
            col_similarity_info = SimilarityInfo(
                score=overall_column_similarity_score,
                prediction_factor_confidence=col_prediction_factor_confidence,
            )
            column_similarity_scores[(column1_id, column2_id)] = col_similarity_info
    logger.info("===============================================")
    return table_similarity_score, column_similarity_scores


def generate_embeddings(table_info_list: List[TableInfo]) -> List[TableInfo]:
    logger.info(f"** Generating Embeddings **")
    try:
        all_strings = []
        for table_info in table_info_list:
            logger.info(f"** Generating Embeddings for {table_info.metadata.table_id}")
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
        all_strings_with_embeddings = {key: value for key, value in zip(all_strings, all_embeddings)}

        for table_info in table_info_list:
            if table_info.metadata.name:
                table_info.metadata.name_embedding.append(TextEmbeddings('sentence_transformer',
                                                                     all_strings_with_embeddings[
                                                                         table_info.metadata.name.lower().strip()]))
            if table_info.metadata.description:
                table_info.metadata.desc_embedding.append(TextEmbeddings('sentence_transformer',
                                                                     all_strings_with_embeddings[
                                                                         table_info.metadata.description.lower().strip()]))
            for col_info in table_info.column_infos:
                if col_info.metadata.name:
                    col_info.metadata.name_embedding.append(TextEmbeddings('sentence_transformer',
                                                                       all_strings_with_embeddings[
                                                                           col_info.metadata.name.lower().strip()]))
                if col_info.metadata.description:
                    col_info.metadata.desc_embedding.append(TextEmbeddings('sentence_transformer',
                                                                       all_strings_with_embeddings[
                                                                           col_info.metadata.description.lower().strip()]))
    except Exception as e:
        logger.error(f"Failed to Generate Embeddings due to", exc_info=e)
    return table_info_list
