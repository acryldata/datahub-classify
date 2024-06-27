import logging
import re
from typing import Any, Dict, List, Union

from datahub_classify.constants import (
    EXCLUDE_NAME,
    PREDICTION_FACTORS_AND_WEIGHTS,
    VALUES,
)
from datahub_classify.helper_classes import Metadata

logger = logging.getLogger(__name__)


def strip_formatting(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


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
    metadata: Metadata,
    values: List[Any],
    config_dict: Dict[str, Union[Dict, List[str], None]],
    infotype: str,
    minimum_values_threshold: int,
) -> bool:
    basic_checks_status = True
    metadata.name = (
        metadata.name
        if not config_dict.get("strip_formatting")
        else strip_formatting(metadata.name)
    )
    prediction_factors = config_dict.get(PREDICTION_FACTORS_AND_WEIGHTS)
    exclude_name = config_dict.get(EXCLUDE_NAME, [])
    if (
        isinstance(prediction_factors, dict)
        and prediction_factors.get(VALUES, None)
        and len(values) < minimum_values_threshold
    ):
        logger.debug(
            f"The number of values for column {metadata.name} "
            f"does not meet minimum threshold for {infotype}"
        )
        basic_checks_status = False
    elif exclude_name is not None and metadata.name in exclude_name:
        logger.debug(f"Excluding match for {infotype} on column {metadata.name}")
        basic_checks_status = False
    # TODO: Add more basic checks
    return basic_checks_status
