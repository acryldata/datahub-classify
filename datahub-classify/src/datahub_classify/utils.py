import logging
import os
import re
from typing import Any, Dict, List, Optional

import nltk
import numpy as np
from numpy.linalg import norm

# from sentence_transformers import SentenceTransformer
from thefuzz import fuzz

from datahub_classify.constants import PREDICTION_FACTORS_AND_WEIGHTS, VALUES
from datahub_classify.helper_classes import ColumnMetadata, TextEmbeddings

logger = logging.getLogger(__name__)
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


# Load Stopwords
def load_stopwords():
    try:
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("english"))

    except Exception as e:
        logger.debug(
            f"Could not Load Stopwords due to {e}: Downloading Stopwords......."
        )
        nltk.download("stopwords")
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("english"))
    return stop_words


# model = SentenceTransformer("all-MiniLM-L6-v2")


# TODO: Exception handling
# Match regex for Name and Description
def match_regex(text_to_match: str, regex_list: List[str]) -> float:
    original_text = text_to_match.lower()
    cleaned_text = "".join(e for e in original_text if e.isalpha())
    match_score: float = 0
    for pattern in regex_list:
        try:
            # TODO: evaluate a case if [A-Za-z] is present in the pattern then it will not give any error,
            # TODO: are there any other cases like above?
            pattern = pattern.lower()
            cleaned_pattern = "".join(e for e in pattern if e.isalpha())
            if (cleaned_pattern == cleaned_text) or (
                re.fullmatch(pattern, original_text)
            ):
                match_score = 1
                break
            # elif re.match(pattern,cleaned_text):  ## revisit later
            #     match_score = 1
            #     break
            elif pattern in original_text:
                match_score = 0.65
            else:
                pass
        except Exception as e:
            logger.error(f"Column Name matching failed due to: {e}")
    return match_score


# Match data type
def match_datatype(dtype_to_match: str, dtype_list: List[str]) -> int:
    dtype_list = [str(s).lower() for s in dtype_list]
    dtype_to_match = dtype_to_match.lower()
    if dtype_to_match in dtype_list:
        match_score = 1
    else:
        match_score = 0
    return match_score


# Match regex for values
def match_regex_for_values(values: List[Any], regex_list: List[str]) -> float:
    values_score_list = []
    length_values = len(values)
    values = [str(x).lower() for x in values]
    for pattern in regex_list:
        try:
            r = re.compile(pattern)
            matches = list(filter(r.fullmatch, values))
            values = [val for val in values if val not in matches]
            values_score_list.append(len(matches))
            if len(values) == 0:
                break
        except Exception as e:
            # TODO: print the exception for debugging purpose
            logger.error(f"Regex match for values failed due to: {e}", exc_info=e)
    values_score = sum(values_score_list) / length_values
    return values_score


def detect_named_entity_spacy(
    spacy_models_list: List, entities_of_interest: List[str], value: str
) -> bool:
    for spacy_model in spacy_models_list:
        doc = spacy_model(value)
        for ent in doc.ents:
            if ent.label_ in entities_of_interest:
                return True
    return False


def perform_basic_checks(
    metadata: ColumnMetadata,
    values: List[Any],
    config_dict: Dict[str, Dict],
    infotype: Optional[str] = None,
) -> bool:
    basic_checks_status = True
    minimum_values_threshold = 50
    if (
        config_dict[PREDICTION_FACTORS_AND_WEIGHTS].get(VALUES, None)
        and len(values) < minimum_values_threshold
    ):
        basic_checks_status = False
    # TODO: Add more basic checks
    return basic_checks_status


def cosine_similarity_score(vec1, vec2):
    try:
        cos_sim = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    except ValueError as e:
        logger.error(f"Failed to get cosine similarity - \n {str(e)}")
    if cos_sim <= 0:
        cos_sim = 0
    return cos_sim


def read_glove_vector(glove_vector: str) -> dict:
    with open(glove_vector, "r", encoding="UTF-8") as f:
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
    return word_to_vec_map


# TODO:
def fuzzy_score_calculation():
    pass


# TODO:
def embedding_score_calculation():
    pass


def compute_string_similarity(
    text_1: Optional[str],
    text_2: Optional[str],
    text_1_emb: List[TextEmbeddings],
    text_2_emb: List[TextEmbeddings],
    text_type: str,
    word_to_vec_map: dict,
    stop_words: set,
    use_embeddings: bool,
) -> Optional[float]:
    try:
        emb_match_score = 0.0
        if text_1 is not None and text_1 != "" and text_2 is not None and text_2 != "":
            # Text pre Processing
            text_1 = text_1.lower().strip()
            text_2 = text_2.lower().strip()
            text_1_cleaned = re.sub(r"[^a-z]", " ", text_1.lower()).strip()
            text_2_cleaned = re.sub(r"[^a-z]", " ", text_2.lower()).strip()
            text_1_words = [
                word for word in text_1_cleaned.split() if word not in stop_words
            ]
            text_2_words = [
                word for word in text_2_cleaned.split() if word not in stop_words
            ]
            col1_emb_type: str = "None"
            col2_emb_type: str = "None"
            # Calculate Embedding Score
            if use_embeddings and text_1_emb is not None and text_2_emb is not None:
                emb_1 = None
                emb_2 = None
                for text_emb in text_1_emb:
                    if text_emb.emb_type == "sentence_transformer":
                        emb_1 = text_emb.embedding
                        col1_emb_type = "SENTENCE"
                        break
                for text_emb in text_2_emb:
                    if text_emb.emb_type == "sentence_transformer":
                        emb_2 = text_emb.embedding
                        col2_emb_type = "SENTENCE"
                        break
                if len(text_1_words) == 1 and len(text_2_words) == 1:
                    glove_emb_1 = word_to_vec_map.get(text_1_words[0], None)
                    glove_emb_2 = word_to_vec_map.get(text_2_words[0], None)

                    if glove_emb_1 is not None and glove_emb_2 is not None:
                        emb_1 = glove_emb_1
                        emb_2 = glove_emb_2
                        col1_emb_type = "GLOVE"
                        col2_emb_type = "GLOVE"
                if emb_1 is None or emb_2 is None:
                    raise Exception("Embeddings not found!!!")
                emb_match_score = cosine_similarity_score(emb_1, emb_2)

            assigned_embedding = [col1_emb_type, col2_emb_type]
            logger.debug(
                f"Found Embeddings: {assigned_embedding} for pair {text_1} and {text_2}"
            )
            # Calculate fuzzy score
            # TODO: Use function to calculate fuzzy score for cleanliness of the script
            fuzzy_match_score = fuzz.token_set_ratio(text_1, text_2) / 100
            if fuzzy_match_score <= 0.5:
                fuzzy_match_score = 0.8 * fuzzy_match_score
            max_fuzz_score = 0
            if text_type == "name":
                if len(text_1_words) == 1 or len(text_2_words) == 1:
                    for word_1 in text_1_words:
                        for word_2 in text_2_words:
                            fuzz_score = fuzz.token_set_ratio(word_1, word_2) / 100
                            if fuzz_score > max_fuzz_score:
                                max_fuzz_score = fuzz_score
            fuzzy_match_score = np.maximum(fuzzy_match_score, max_fuzz_score)
            score = np.maximum(fuzzy_match_score, emb_match_score)
        else:
            score = None
    except Exception as e:
        logger.error(
            f"Failed to find name / description similarity for texts: '{text_1}' and '{text_2}'",
            exc_info=e,
        )
        score = None
    return score


def download_glove_embeddings(glove_vec):
    try:
        destination_path, glove_file = os.path.split(glove_vec)
        from io import BytesIO
        from zipfile import ZipFile

        import requests

        # URL = "https://nlp.stanford.edu/data/glove.6B.zip"
        response = requests.get(GLOVE_URL)
        zip_object = ZipFile(BytesIO(response.content))
        zip_object.extract(glove_file, path=destination_path)
        logger.debug("Successfully Downloaded GLOVE Embeddings!!")
    except Exception as e:
        logger.error(f"Unable To Download GLOVE Embeddings due to {e}", exc_info=e)
