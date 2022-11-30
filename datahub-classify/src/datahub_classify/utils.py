import logging
import re

import numpy as np

# GLOVE URL :  https://nlp.stanford.edu/data/glove.6B.zip
from nltk.corpus import stopwords
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from thefuzz import fuzz

from datahub_classify.constants import PREDICTION_FACTORS_AND_WEIGHTS, VALUES

logger = logging.getLogger(__name__)
stop_words = set(stopwords.words("english"))

model = SentenceTransformer("all-MiniLM-L6-v2")


# Match regex for Name and Description
def match_regex(text_to_match, regex_list):
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
def match_datatype(dtype_to_match, dtype_list):
    dtype_list = [str(s).lower() for s in dtype_list]
    dtype_to_match = dtype_to_match.lower()
    if dtype_to_match in dtype_list:
        match_score = 1
    else:
        match_score = 0
    return match_score


# Match regex for values
def match_regex_for_values(values, regex_list):
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
            logger.error("Regex match for values failed due to: ", exc_info=e)
    values_score = sum(values_score_list) / length_values
    return values_score


def detect_named_entity_spacy(spacy_models_list, entities_of_interest, value):
    for spacy_model in spacy_models_list:
        doc = spacy_model(value)
        for ent in doc.ents:
            if ent.label_ in entities_of_interest:
                return True
    return False


def perform_basic_checks(metadata, values, config_dict, infotype=None):
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


# TODO: Rename 'name_dsc_similarity' to compute_string_similarity  and move the similarity specific functions to infotype predictor
def compute_string_similarity(
    text_1: str, text_2: str, text_type: str, word_to_vec_map: dict
) -> float:
    try:
        text_1 = text_1.lower().strip()
        text_2 = text_2.lower().strip()

        if (text_1 not in ([None, ""])) and (text_2 not in ([None, ""])):
            fuzzy_match_score = fuzz.token_set_ratio(text_1, text_2) / 100
            if fuzzy_match_score <= 0.5:
                fuzzy_match_score = 0.8 * fuzzy_match_score

            text_1_cleaned = re.sub(r"[^a-z]", " ", text_1.lower()).strip()
            text_2_cleaned = re.sub(r"[^a-z]", " ", text_2.lower()).strip()
            text_1_words = [
                word for word in text_1_cleaned.split() if word not in (stop_words)
            ]
            text_2_words = [
                word for word in text_2_cleaned.split() if word not in (stop_words)
            ]

            max_fuzz_score = 0
            if text_type == "name":
                if len(text_1_words) == 1 or len(text_2_words) == 1:
                    for word_1 in text_1_words:
                        for word_2 in text_2_words:
                            fuzz_score = fuzz.token_set_ratio(word_1, word_2) / 100
                            if fuzz_score > max_fuzz_score:
                                max_fuzz_score = fuzz_score

            fuzzy_match_score = np.maximum(fuzzy_match_score, max_fuzz_score)

            # TODO: do we need "<=" or "==" in the following condition?
            if len(text_1_words) == 1 and len(text_2_words) == 1:
                # TODO: can we change following two statements to "emb_x = word_to_vec_map[text_x_words[0]"
                # TODO: only one word is present in "text_x_words" so we don't need list comprehension
                emb_1 = word_to_vec_map.get(text_1_words[0], "nan")
                emb_2 = word_to_vec_map.get(text_2_words[0], "nan")

                if str(emb_1) == "nan" or str(emb_2) == "nan":
                    embeddings = model.encode([text_1, text_2], show_progress_bar=False)
                    emb_1 = embeddings[0]
                    emb_2 = embeddings[1]
            else:
                embeddings = model.encode([text_1, text_2], show_progress_bar=False)
                emb_1 = embeddings[0]
                emb_2 = embeddings[1]
            emb_match_score = cosine_similarity_score(emb_1, emb_2)
            score = np.maximum(fuzzy_match_score, emb_match_score)
        else:
            score = 0
    except Exception as e:
        logger.error(
            f"Failed to find name / description similarity for texts: '{text_1}' and '{text_2}'",
            exc_info=e,
        )

        # print(text_1)
        # print(text_2)
        # print("fuzzy score: ", fuzzy_match_score)
        # print("glove score: ", emb_match_score)
    return score
