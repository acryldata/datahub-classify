import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

import datahub_classify.similarity_numeric_constants as config
from datahub_classify.helper_classes import (
    ColumnInfo,
    ScoreInfo,
    SimilarityFactorScoreInfo,
    SimilarityInfo,
    TableInfo,
    TextEmbeddings,
)
from datahub_classify.utils import (
    compute_string_similarity,
    get_fuzzy_score,
    load_stopwords,
    read_glove_vector,
)

logger = logging.getLogger(__name__)
current_wdr = os.path.dirname(os.path.abspath(__file__))
glove_vec = os.path.join(current_wdr, "glove.6B.50d.txt")
stop_words = load_stopwords()

try:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logger.exception(f"Failed to load sentence transformer due to {e}")
    model = None


if not os.path.isfile(glove_vec):
    from datahub_classify.utils import download_glove_embeddings

    logger.debug("Downloading GLOVE Embeddings.............")
    download_glove_embeddings(glove_vec)

word_to_vec_map = read_glove_vector(glove_vec)


def column_dtype_similarity(
    column_1_dtype: Optional[str], column_2_dtype: Optional[str]
) -> Optional[int]:
    try:
        if column_1_dtype == column_2_dtype:
            column_dtype_score = 1
        else:
            column_dtype_score = 0
    except Exception as e:
        logger.exception(
            f"Failed to compute column dtype similarity for '{column_1_dtype}' and '{column_2_dtype}' due to {e}"
        )
        column_dtype_score = None
    return column_dtype_score


def table_platform_similarity(
    platform_1_name: Optional[str], platform_2_name: Optional[str]
) -> Optional[int]:
    try:
        if platform_1_name != platform_2_name:
            platform_score = 1
        else:
            platform_score = 0
    except Exception as e:
        logger.exception(f"Failed to compute platform score due to {e}")
        platform_score = None
    return platform_score


def compute_lineage_score(
    entity_1_parents: List,
    entity_2_parents: List,
    entity_1_id: Optional[str],
    entity_2_id: Optional[str],
) -> Optional[int]:
    try:
        if (entity_1_id in entity_2_parents) or (entity_2_id in entity_1_parents):
            lineage_score = 1
        else:
            lineage_score = 0
    except Exception as e:
        logger.exception(
            f"Failed to compute lineage score for entities with IDs: '{entity_1_id}' and '{entity_2_id}' due to {e}"
        )
        lineage_score = None
    return lineage_score


def table_schema_similarity(
    col_infos1: List[ColumnInfo],
    col_infos2: List[ColumnInfo],
    use_embeddings: bool,
    pair_score_threshold: float = config.SCHEMA_COL_PAIR_SCORE_THRESHOLD,
) -> Optional[float]:
    try:
        num_matched_pairs = 0
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
                    word_to_vec_map=word_to_vec_map,
                    use_embeddings=use_embeddings,
                    stop_words=stop_words,
                )
                col_dtype_score = column_dtype_similarity(
                    col1.metadata.datatype, col2.metadata.datatype
                )
                if col_name_score is not None and col_dtype_score is not None:
                    pair_score = (
                        config.SCHEMA_COL_NAME_WEIGHT * col_name_score
                        + config.SCHEMA_COL_DTYPE_WEIGHT * col_dtype_score
                    )

                    if pair_score > pair_score_threshold:
                        num_matched_pairs += 1
                        columns_to_remove.append(col2.metadata.column_id)
                        break
        schema_score = 2 * num_matched_pairs / (num_cols_table1 + num_cols_table2)
    except Exception as e:
        logger.exception(f"Failed to compute table schema similarity due to {e}")
        schema_score = None
    return schema_score


def table_schema_similarity_pruning(
    col_infos1: List[ColumnInfo],
    col_infos2: List[ColumnInfo],
) -> Optional[float]:
    try:
        text_1 = ""
        text_2 = ""

        assert col_infos1 and col_infos2
        assert len(col_infos1) > 0 and len(col_infos2) > 0

        for col in col_infos1:
            dtype = col.metadata.datatype
            name = col.metadata.name
            if (
                name is not None
                and dtype is not None
                and name.strip() != ""
                and dtype != ""
            ):
                if not name.isdigit():
                    name = re.sub(
                        r"[^a-zA-Z0-9]+",
                        "_",
                        re.sub(r"^[^a-zA-Z]+", "", name.lower().strip()),
                    )
                text_1 = text_1 + " " + f"{name}_{dtype}"

        for col in col_infos2:
            dtype = str(col.metadata.datatype)
            name = col.metadata.name
            if (
                name is not None
                and dtype is not None
                and name.strip() != ""
                and dtype != ""
            ):
                if not name.isdigit():
                    name = re.sub(
                        r"[^a-zA-Z0-9]+",
                        "_",
                        re.sub(r"^[^a-zA-Z]+", "", name.lower().strip()),
                    )
                text_2 = text_2 + " " + f"{name}_{dtype}"
        if text_1 != "" and text_2 != "":
            schema_score = get_fuzzy_score(
                text_1=text_1, text_2=text_2, text_type="schema"
            )
        else:
            schema_score = None
    except Exception as e:
        logger.exception(f"Failed to compute table schema similarity due to {e}")
        schema_score = None
    return schema_score


def compute_table_overall_similarity_score(
    name_score: float,
    desc_score: Optional[float],
    platform_score: Optional[int],
    lineage_score: Optional[int],
    schema_score: float,
) -> Tuple[float, Dict[str, Optional[float]]]:
    weighted_factor_scores: Dict[str, Optional[float]] = dict()

    # Name Score
    weighted_factor_scores["name"] = config.OVERALL_NAME_SCORE_WEIGHT * name_score
    # Schema Score
    weighted_factor_scores["table_schema"] = (
        config.OVERALL_SCHEMA_SCORE_WEIGHT * schema_score
    )

    # Platform Score
    if platform_score is None:
        weighted_factor_scores["platform"] = None
    else:
        weighted_factor_scores["platform"] = (
            config.OVERALL_PLATFORM_SCORE_WEIGHT * platform_score
        )

    weighted_score = sum(filter(None, weighted_factor_scores.values()))

    # Description Score
    if desc_score is not None:
        if desc_score >= config.TABLE_DESC_THRESHOLD:
            weighted_score = np.minimum(config.OVERALL_DESC_BOOST * weighted_score, 1)
        if desc_score < config.OVERALL_DESC_SCORE_THRESHOLD:
            weighted_score = config.OVERALL_DESC_PENALTY * weighted_score
        weighted_factor_scores["description"] = weighted_score - sum(
            filter(None, weighted_factor_scores.values())
        )
    else:
        weighted_factor_scores["description"] = None

    overall_table_similarity_score = weighted_score

    # Lineage Score
    if (
        weighted_score >= config.TABLE_WEIGHTED_SCORE_THRESHOLD
        and lineage_score is not None
        and lineage_score == 1
    ):
        overall_table_similarity_score = 1
        weighted_factor_scores["lineage"] = 1.0
    else:
        weighted_factor_scores["lineage"] = 0.0
    return np.round(overall_table_similarity_score, 2), weighted_factor_scores


def compute_column_overall_similarity_score(
    name_score: float,
    dtype_score: int,
    desc_score: Optional[float],
    table_similarity_score: Optional[float],
    lineage_score: Optional[int],
) -> Tuple[float, Dict[str, Optional[float]]]:
    weighted_factor_scores: Dict[str, Optional[float]] = dict()
    # Name Score
    weighted_factor_scores["name"] = name_score

    # Datatype Score
    if dtype_score == 1:
        weighted_score = config.COLUMN_DTYPE_BOOST * name_score
    else:
        weighted_score = config.COLUMN_DTYPE_PENALTY * name_score
    weighted_factor_scores["datatype"] = weighted_score - name_score

    # Description Score
    if desc_score is not None:
        if desc_score >= config.COLUMN_DESC_THRESHOLD:
            weighted_score = config.COLUMN_DESC_BOOST * weighted_score
        if desc_score < config.OVERALL_DESC_SCORE_THRESHOLD:
            weighted_score = config.COLUMN_DESC_PENALTY * weighted_score
        weighted_factor_scores["description"] = weighted_score - sum(
            filter(None, weighted_factor_scores.values())
        )
    else:
        weighted_factor_scores["description"] = None

    # Schema Score & Lineage Score
    if weighted_score > config.COLUMN_WEIGHTED_SCORE_THRESHOLD:
        if (
            table_similarity_score is not None
            and table_similarity_score > config.TABLE_SIMILARITY_THRESHOLD
        ):
            weighted_score = config.COLUMN_TABLE_SIMILARITY_BOOST * weighted_score
            weighted_factor_scores["table_schema"] = weighted_score - sum(
                filter(None, weighted_factor_scores.values())
            )
        else:
            weighted_factor_scores["table_schema"] = 0.0
        if lineage_score is not None and lineage_score == 1:
            weighted_score = config.COLUMN_LINEAGE_BOOST * weighted_score
            weighted_factor_scores["lineage"] = weighted_score - sum(
                filter(None, weighted_factor_scores.values())
            )
        else:
            weighted_factor_scores["lineage"] = 0.0
    else:
        weighted_factor_scores["table_schema"] = None
        weighted_factor_scores["lineage"] = None

    overall_column_similarity_score = np.minimum(weighted_score, 1)
    return np.round(overall_column_similarity_score, 2), weighted_factor_scores


def compute_table_similarity(
    table_info1: TableInfo,
    table_info2: TableInfo,
    use_embeddings: bool,
    pruning_mode: bool,
) -> Tuple[Optional[float], Optional[SimilarityFactorScoreInfo]]:
    table_name_score = compute_string_similarity(
        table_info1.metadata.name,
        table_info2.metadata.name,
        table_info1.metadata.name_embedding,
        table_info2.metadata.name_embedding,
        text_type="name",
        word_to_vec_map=word_to_vec_map,
        use_embeddings=use_embeddings,
        stop_words=stop_words,
    )
    if (
        table_info1.metadata.description
        and table_info2.metadata.description
        and table_info1.metadata.description.strip() != ""
        and table_info2.metadata.description.strip() != ""
    ):
        table_desc_score = compute_string_similarity(
            table_info1.metadata.description,
            table_info2.metadata.description,
            table_info1.metadata.desc_embedding,
            table_info2.metadata.desc_embedding,
            text_type="desc",
            word_to_vec_map=word_to_vec_map,
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
    if pruning_mode:
        table_schema_score = table_schema_similarity_pruning(
            table_info1.column_infos,
            table_info2.column_infos,
        )
    else:
        table_schema_score = table_schema_similarity(
            table_info1.column_infos,
            table_info2.column_infos,
            use_embeddings=use_embeddings,
        )

    if table_name_score is None or table_schema_score is None:
        overall_table_similarity_score = None
        table_prediction_factor_scores = None
    else:
        (
            overall_table_similarity_score,
            weighted_factor_scores,
        ) = compute_table_overall_similarity_score(
            table_name_score,
            table_desc_score,
            table_platform_score,
            table_lineage_score,
            table_schema_score,
        )
        table_prediction_factor_scores = SimilarityFactorScoreInfo(
            name=ScoreInfo(
                confidence=table_name_score,
                weighted_score=weighted_factor_scores["name"],
            ),
            description=ScoreInfo(
                confidence=table_desc_score,
                weighted_score=weighted_factor_scores["description"],
            ),
            platform=ScoreInfo(
                confidence=table_platform_score,
                weighted_score=weighted_factor_scores["platform"],
            ),
            lineage=ScoreInfo(
                confidence=table_lineage_score,
                weighted_score=weighted_factor_scores["lineage"],
            ),
            table_schema=ScoreInfo(
                confidence=table_schema_score,
                weighted_score=weighted_factor_scores["table_schema"],
            ),
        )
    return overall_table_similarity_score, table_prediction_factor_scores


def compute_column_similarity(
    col_info1: ColumnInfo,
    col_info2: ColumnInfo,
    overall_table_similarity_score: Optional[float],
    use_embeddings: bool,
) -> Tuple[Optional[float], Optional[SimilarityFactorScoreInfo]]:
    column_name_score = compute_string_similarity(
        col_info1.metadata.name,
        col_info2.metadata.name,
        col_info1.metadata.name_embedding,
        col_info2.metadata.name_embedding,
        text_type="name",
        word_to_vec_map=word_to_vec_map,
        use_embeddings=use_embeddings,
        stop_words=stop_words,
    )

    if (
        col_info1.metadata.description
        and col_info2.metadata.description
        and col_info1.metadata.description.strip() != ""
        and col_info2.metadata.description.strip() != ""
    ):
        column_desc_score = compute_string_similarity(
            col_info1.metadata.description,
            col_info2.metadata.description,
            col_info1.metadata.desc_embedding,
            col_info2.metadata.desc_embedding,
            text_type="desc",
            word_to_vec_map=word_to_vec_map,
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

    if column_name_score is None or column_dtype_score is None:
        overall_column_similarity_score = None
        col_prediction_factor_confidence = None
    else:
        (
            overall_column_similarity_score,
            weighted_factor_scores,
        ) = compute_column_overall_similarity_score(
            column_name_score,
            column_dtype_score,
            column_desc_score,
            overall_table_similarity_score,
            column_lineage_score,
        )
        col_prediction_factor_confidence = SimilarityFactorScoreInfo(
            name=ScoreInfo(
                confidence=column_name_score,
                weighted_score=weighted_factor_scores["name"],
            ),
            description=ScoreInfo(
                confidence=column_desc_score,
                weighted_score=weighted_factor_scores["description"],
            ),
            datatype=ScoreInfo(
                confidence=column_dtype_score,
                weighted_score=weighted_factor_scores["datatype"],
            ),
            lineage=ScoreInfo(
                confidence=column_lineage_score,
                weighted_score=weighted_factor_scores["lineage"],
            ),
            table_schema=ScoreInfo(
                confidence=overall_table_similarity_score,
                weighted_score=weighted_factor_scores["table_schema"],
            ),
        )

    return overall_column_similarity_score, col_prediction_factor_confidence


def check_similarity(
    table_info1: TableInfo,
    table_info2: TableInfo,
    pruning_mode: bool = False,
    use_embeddings: bool = True,
) -> tuple:
    logger.debug(
        f"** Finding table similarity between Table '{table_info1.metadata.table_id}' and '{table_info2.metadata.table_id}' **"
    )
    try:
        (
            overall_table_similarity_score,
            table_prediction_factor_confidence,
        ) = compute_table_similarity(
            table_info1, table_info2, use_embeddings, pruning_mode
        )
    except Exception as e:
        logger.exception(
            f"Failed to compute table similarity between Table "
            f"{table_info1.metadata.table_id} and {table_info2.metadata.table_id} due to {e}"
        )
        overall_table_similarity_score = None
        table_prediction_factor_confidence = None

    table_similarity_score = SimilarityInfo(
        score=overall_table_similarity_score,
        prediction_factors_scores=table_prediction_factor_confidence,
    )

    column_similarity_scores: Dict[tuple, SimilarityInfo] = {}

    if not pruning_mode:
        logger.debug(
            f"Total pairs --> {len(table_info1.column_infos) * len(table_info2.column_infos)}"
        )
        for col_info1 in table_info1.column_infos:
            for col_info2 in table_info2.column_infos:
                try:
                    if (
                        overall_table_similarity_score is not None
                        and overall_table_similarity_score
                        > config.OVERALL_TABLE_SIMILARITY_THRESHOLD
                    ):
                        if col_info1.metadata.datatype == col_info2.metadata.datatype:
                            (
                                overall_column_similarity_score,
                                col_prediction_factor_confidence,
                            ) = compute_column_similarity(
                                col_info1,
                                col_info2,
                                overall_table_similarity_score,
                                use_embeddings,
                            )
                        else:
                            overall_column_similarity_score = 0
                            col_prediction_factor_confidence = (
                                SimilarityFactorScoreInfo(
                                    datatype=ScoreInfo(confidence=0, weighted_score=0)
                                )
                            )
                    else:
                        overall_column_similarity_score = 0
                        col_prediction_factor_confidence = SimilarityFactorScoreInfo()
                except Exception as e:
                    logger.exception(
                        f"Failed to compute column similarity between Column "
                        f"{col_info1.metadata.column_id} and {col_info1.metadata.column_id} due to {e}"
                    )
                    overall_column_similarity_score = None
                    col_prediction_factor_confidence = None

                column1_id = col_info1.metadata.column_id
                column2_id = col_info2.metadata.column_id
                col_similarity_info = SimilarityInfo(
                    score=overall_column_similarity_score,
                    prediction_factors_scores=col_prediction_factor_confidence,
                )
                column_similarity_scores[(column1_id, column2_id)] = col_similarity_info
    # logger.info("===============================================")
    return table_similarity_score, column_similarity_scores


def preprocess_tables(table_info_list: List[TableInfo]) -> List[TableInfo]:
    try:
        all_strings = []
        for table_info in table_info_list:
            if table_info.metadata.name:
                all_strings.append(table_info.metadata.name)
            if table_info.metadata.description:
                all_strings.append(table_info.metadata.description)
            for col_info in table_info.column_infos:
                if col_info.metadata.name:
                    all_strings.append(col_info.metadata.name)
                if col_info.metadata.description:
                    all_strings.append(col_info.metadata.description)
        all_strings = list(map(lambda x: x.lower().strip(), all_strings))
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
        logger.exception(f"Failed to Generate Embeddings due to {e}")
    return table_info_list
