import ipaddress
import logging
import re
import string
from typing import Any, Dict

import numpy as np
import phonenumbers
import spacy
from schwifty import BIC, IBAN
from stdnum.us import ssn as us_ssn
from vininfo import Vin

from datahub_classify.constants import (
    DATATYPE,
    DESCRIPTION,
    NAME,
    PREDICTION_FACTORS_AND_WEIGHTS,
    PREDICTION_TYPE,
    REGEX,
    TYPE,
    VALUES,
)
from datahub_classify.infotype_utils import (
    detect_named_entity_spacy,
    match_datatype,
    match_regex,
    match_regex_for_values,
)

logger = logging.getLogger(__name__)
spacy_model_name = "en_core_web_sm"
try:
    nlp_english = spacy.load(spacy_model_name)
except OSError:
    logger.info("Downloading language model for the spaCy")
    from spacy.cli import download

    download(spacy_model_name)
    nlp_english = spacy.load(spacy_model_name)
spacy_models_list = [nlp_english]

# TODO: pull out the common/repetitive code


def inspect_for_email_address(metadata, values, config):
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info: Dict[str, Any] = {}

    # Value Logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0
        try:
            if config[VALUES][PREDICTION_TYPE] == "regex":
                values_score = match_regex_for_values(values, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == "library":
                raise Exception(
                    "Currently prediction type 'library' is not supported for infotype Email Address"
                )
            else:
                raise Exception(
                    f"Inappropriate Prediction type {config[VALUES][PREDICTION_TYPE]}"
                )
        except Exception as e:
            logger.error(f"Column {metadata.name} failed due to {e}")
        values_score = np.round(values_score, 2)
        debug_info[VALUES] = values_score

    # Name Logic
    if prediction_factors_weights.get(NAME, 0) > 0:
        if not metadata.name or not metadata.name.strip():
            debug_info[NAME] = f"0.0 (Blank {NAME} Metadata)"
        else:
            debug_info[NAME] = match_regex(metadata.name, config[NAME][REGEX])

    # Description_Logic
    if prediction_factors_weights.get(DESCRIPTION, 0) > 0:
        if not metadata.description or not metadata.description.strip():
            debug_info[DESCRIPTION] = f"0.0 (Blank {DESCRIPTION} Metadata)"
        else:
            debug_info[DESCRIPTION] = match_regex(
                metadata.description, config[DESCRIPTION][REGEX]
            )

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE] = match_datatype(
                metadata.datatype, config[DATATYPE][TYPE]
            )

    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key]) != str:
            confidence_level += prediction_factors_weights[key] * debug_info[key]
    confidence_level = np.round(confidence_level, 2)
    return confidence_level, debug_info


def inspect_for_street_address(metadata, values, config):  # noqa: C901
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info: Dict[str, Any] = {}

    # Values logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0
        try:
            if config[VALUES][PREDICTION_TYPE] == "regex":
                values_score = match_regex_for_values(values, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == "library":
                entity_count: float = 0
                entities_of_interest = ["FAC", "LOC", "ORG"]
                weight = 1.5
                for value in values:
                    try:
                        if detect_named_entity_spacy(
                            spacy_models_list, entities_of_interest, value
                        ):
                            entity_count += weight
                    except Exception:
                        pass
                entities_score = entity_count / len(values)
                values_score = np.minimum(entities_score, 1)
            else:
                raise Exception(
                    f"Inappropriate values_prediction_type {config[VALUES][PREDICTION_TYPE]}"
                )
        except Exception as e:
            logger.error(f"Column {metadata.name} failed due to {e}")
            pass
        values_score = np.round(values_score, 2)
        debug_info[VALUES] = values_score

    # Name Logic
    if prediction_factors_weights.get(NAME, 0) > 0:
        if not metadata.name or not metadata.name.strip():
            debug_info[NAME] = f"0.0 (Blank {NAME} Metadata)"
        else:
            debug_info[NAME] = match_regex(metadata.name, config[NAME][REGEX])

    # Description_Logic
    if prediction_factors_weights.get(DESCRIPTION, 0) > 0:
        if not metadata.description or not metadata.description.strip():
            debug_info[DESCRIPTION] = f"0.0 (Blank {DESCRIPTION} Metadata)"
        else:
            debug_info[DESCRIPTION] = match_regex(
                metadata.description, config[DESCRIPTION][REGEX]
            )

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE] = match_datatype(
                metadata.datatype, config[DATATYPE][TYPE]
            )

    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key]) != str:
            confidence_level += prediction_factors_weights[key] * debug_info[key]
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_gender(metadata, values, config):  # noqa: C901
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info: Dict[str, Any] = {}

    # Value Logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0
        try:
            if config[VALUES][PREDICTION_TYPE] == "regex":
                values_score = match_regex_for_values(values, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == "library":
                raise Exception(
                    "Currently prediction type 'library' is not supported for infotype Gender"
                )
            else:
                raise Exception(
                    f"Inappropriate values_prediction_type {config[VALUES][PREDICTION_TYPE]}"
                )
        except Exception as e:
            logger.error(f"Column {metadata.name} failed due to {e}")
        values_score = np.round(values_score, 2)
        debug_info[VALUES] = values_score

    # Name Logic
    if prediction_factors_weights.get(NAME, 0) > 0:
        if not metadata.name or not metadata.name.strip():
            debug_info[NAME] = f"0.0 (Blank {NAME} Metadata)"
        else:
            debug_info[NAME] = match_regex(metadata.name, config[NAME][REGEX])

    # Description_Logic
    if prediction_factors_weights.get(DESCRIPTION, 0) > 0:
        if not metadata.description or not metadata.description.strip():
            debug_info[DESCRIPTION] = f"0.0 (Blank {DESCRIPTION} Metadata)"
        else:
            debug_info[DESCRIPTION] = match_regex(
                metadata.description, config[DESCRIPTION][REGEX]
            )

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE] = match_datatype(
                metadata.datatype, config[DATATYPE][TYPE]
            )
    try:
        if (
            debug_info.get(NAME, None)
            and int(debug_info[NAME]) == 1
            and VALUES in debug_info.keys()
            and debug_info[VALUES] == 0
        ):
            num_unique_values = len(np.unique(values))
            if num_unique_values < 5:
                debug_info[VALUES] = 0.9
    except Exception as e:
        logger.error(
            f"Failed to evaluate unique values whn name_score==1 for Gender due to {e}"
        )

    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key]) != str:
            confidence_level += prediction_factors_weights[key] * debug_info[key]
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_credit_debit_card_number(metadata, values, config):
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info: Dict[str, Any] = {}

    # Value Logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0
        try:
            values = [str(s).lower() for s in values]
            values_cleaned = []
            for value in values:
                string_cleaned = re.sub(r"[ _-]+", "", value)
                values_cleaned.append(string_cleaned)
            if config[VALUES][PREDICTION_TYPE] == "regex":
                values_score = match_regex_for_values(
                    values_cleaned, config[VALUES][REGEX]
                )
            elif config[VALUES][PREDICTION_TYPE] == "library":
                raise Exception(
                    "Currently prediction type 'library' is not supported for infotype Credit Card Number"
                )
            else:
                raise Exception(
                    f"Inappropriate values_prediction_type {config[VALUES][PREDICTION_TYPE]}"
                )
        except Exception as e:
            logger.error(f"Column {metadata.name} failed due to {e}")
        values_score = np.round(values_score, 2)
        debug_info[VALUES] = values_score

    # Name Logic
    if prediction_factors_weights.get(NAME, 0) > 0:
        if not metadata.name or not metadata.name.strip():
            debug_info[NAME] = f"0.0 (Blank {NAME} Metadata)"
        else:
            debug_info[NAME] = match_regex(metadata.name, config[NAME][REGEX])

    # Description_Logic
    if prediction_factors_weights.get(DESCRIPTION, 0) > 0:
        if not metadata.description or not metadata.description.strip():
            debug_info[DESCRIPTION] = f"0.0 (Blank {DESCRIPTION} Metadata)"
        else:
            debug_info[DESCRIPTION] = match_regex(
                metadata.description, config[DESCRIPTION][REGEX]
            )

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE] = match_datatype(
                metadata.datatype, config[DATATYPE][TYPE]
            )

    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key]) != str:
            confidence_level += prediction_factors_weights[key] * debug_info[key]
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_phone_number(metadata, values, config):  # noqa: C901
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info: Dict[str, Any] = {}

    # fmt: off
    # TODO: shall we have these country codes in config?
    iso_codes = ["AF", "AX", "AL", "DZ", "AS", "AD", "AO", "AI", "AQ", "AG", "AR",
                 "AM", "AW", "AU", "AT", "AZ", "BS", "BH", "BD", "BB", "BY", "BE",
                 "BZ", "BJ", "BM", "BT", "BO", "BQ", "BA", "BW", "BV", "BR", "IO",
                 "BN", "BG", "BF", "BI", "CV", "KH", "CM", "CA", "KY", "CF", "TD",
                 "CL", "CN", "CX", "CC", "CO", "KM", "CG", "CD", "CK", "CR", "CI",
                 "HR", "CU", "CW", "CY", "CZ", "DK", "DJ", "DM", "DO", "EC", "EG",
                 "SV", "GQ", "ER", "EE", "ET", "FK", "FO", "FJ", "FI", "FR", "GF",
                 "PF", "TF", "GA", "GM", "GE", "DE", "GH", "GI", "GR", "GL", "GD",
                 "GP", "GU", "GT", "GG", "GN", "GW", "GY", "HT", "HM", "VA", "HN",
                 "HK", "HU", "IS", "IN", "ID", "IR", "IQ", "IE", "IM", "IL", "IT",
                 "JM", "JP", "JE", "JO", "KZ", "KE", "KI", "KP", "KR", "KW", "KG",
                 "LA", "LV", "LB", "LS", "LR", "LY", "LI", "LT", "LU", "MO", "MK",
                 "MG", "MW", "MY", "MV", "ML", "MT", "MH", "MQ", "MR", "MU", "YT",
                 "MX", "FM", "MD", "MC", "MN", "ME", "MS", "MA", "MZ", "MM", "NA",
                 "NR", "NP", "NL", "NC", "NZ", "NI", "NE", "NG", "NU", "NF", "MP",
                 "NO", "OM", "PK", "PW", "PS", "PA", "PG", "PY", "PE", "PH", "PN",
                 "PL", "PT", "PR", "QA", "RE", "RO", "RU", "RW", "BL", "SH", "KN",
                 "LC", "MF", "PM", "VC", "WS", "SM", "ST", "SA", "SN", "RS", "SC",
                 "SL", "SG", "SX", "SK", "SI", "SB", "SO", "ZA", "GS", "SS", "ES",
                 "LK", "SD", "SR", "SJ", "SZ", "SE", "CH", "SY", "TW", "TJ", "TZ",
                 "TH", "TL", "TG", "TK", "TO", "TT", "TN", "TR", "TM", "TC", "TV",
                 "UG", "UA", "AE", "GB", "US", "UM", "UY", "UZ", "VU", "VE", "VN",
                 "VG", "VI", "WF", "EH", "YE", "ZM", "ZW"]
    # fmt: on
    # Values logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score: float = 0
        try:
            if config[VALUES][PREDICTION_TYPE] == "regex":
                values_score = match_regex_for_values(values, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == "library":
                valid_phone_numbers_count = 0
                for value in values:
                    try:
                        for code in iso_codes:
                            parsed_number = phonenumbers.parse(value, code)
                            if phonenumbers.is_possible_number(parsed_number):
                                valid_phone_numbers_count += 1
                                break
                    except Exception:
                        pass
                values_score = valid_phone_numbers_count / len(values)
            else:
                raise Exception(
                    f"Inappropriate values_prediction_type {config[VALUES][PREDICTION_TYPE]}"
                )

        except Exception as e:
            logger.error(f"Column {metadata.name} failed due to {e}")

        values_score = np.round(values_score, 2)
        debug_info[VALUES] = values_score

    # Name Logic
    if prediction_factors_weights.get(NAME, 0) > 0:
        if not metadata.name or not metadata.name.strip():
            debug_info[NAME] = f"0.0 (Blank {NAME} Metadata)"
        else:
            debug_info[NAME] = match_regex(metadata.name, config[NAME][REGEX])

    # Description_Logic
    if prediction_factors_weights.get(DESCRIPTION, 0) > 0:
        if not metadata.description or not metadata.description.strip():
            debug_info[DESCRIPTION] = f"0.0 (Blank {DESCRIPTION} Metadata)"
        else:
            debug_info[DESCRIPTION] = match_regex(
                metadata.description, config[DESCRIPTION][REGEX]
            )

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE] = match_datatype(
                metadata.datatype, config[DATATYPE][TYPE]
            )

    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key]) != str:
            confidence_level += prediction_factors_weights[key] * debug_info[key]
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_full_name(metadata, values, config):  # noqa: C901
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info: Dict[str, Any] = {}

    # Values logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0
        try:
            if config[VALUES][PREDICTION_TYPE] == "regex":
                values_score = match_regex_for_values(values, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == "library":
                entity_count = 0
                entities_of_interest = ["PERSON"]
                weight = 1
                for value in values:
                    try:
                        if (
                            len(value) <= 50
                            and len(re.split(rf"[\s{string.punctuation}]+", value)) >= 2
                        ):
                            if detect_named_entity_spacy(
                                spacy_models_list, entities_of_interest, value
                            ):
                                entity_count += weight
                    except Exception:
                        pass
                entities_score = entity_count / len(values)
                values_score = np.minimum(entities_score, 1)
            else:
                raise Exception(
                    f"Inappropriate values_prediction_type {config[VALUES][PREDICTION_TYPE]}"
                )
        except Exception as e:
            logger.error(f"Column {metadata.name} failed due to {e}")
        values_score = np.round(values_score, 2)
        debug_info[VALUES] = values_score

    # Name Logic
    if prediction_factors_weights.get(NAME, 0) > 0:
        if not metadata.name or not metadata.name.strip():
            debug_info[NAME] = f"0.0 (Blank {NAME} Metadata)"
        else:
            debug_info[NAME] = match_regex(metadata.name, config[NAME][REGEX])

    # Description_Logic
    if prediction_factors_weights.get(DESCRIPTION, 0) > 0:
        if not metadata.description or not metadata.description.strip():
            debug_info[DESCRIPTION] = f"0.0 (Blank {DESCRIPTION} Metadata)"
        else:
            debug_info[DESCRIPTION] = match_regex(
                metadata.description, config[DESCRIPTION][REGEX]
            )

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE] = match_datatype(
                metadata.datatype, config[DATATYPE][TYPE]
            )
    try:
        if (
            debug_info.get(NAME, None)
            and int(debug_info[NAME]) == 1
            and VALUES in debug_info.keys()
            and 0.5 > debug_info[VALUES] > 0.1
        ):
            debug_info[VALUES] = 0.8
    except Exception as e:
        logger.error(f"Column {metadata.name} failed due to {e}")

    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key]) != str:
            confidence_level += prediction_factors_weights[key] * debug_info[key]
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_age(metadata, values, config):  # noqa: C901
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info: Dict[str, Any] = {}

    # Values logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score: float = 0
        try:
            if config[VALUES][PREDICTION_TYPE] == "regex":
                values_score = match_regex_for_values(values, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == "library":
                try:
                    # Check if column is convertible to int dtype
                    int_col = [int(s) for s in values]
                    max_val = np.percentile(int_col, 95)
                    min_val = np.percentile(int_col, 5)
                    num_unique = len(np.unique(int_col))
                    if max_val <= 120 and min_val > 0:
                        values_score += 0.4
                        if num_unique >= np.minimum(len(values) / 10, 40):
                            values_score += 0.3
                    else:
                        values_score = 0
                except Exception:
                    pass
            else:
                raise Exception(
                    f"Inappropriate values_prediction_type {config[VALUES][PREDICTION_TYPE]}"
                )
        except Exception as e:
            logger.error(f"Column {metadata.name} failed due to {e}")
        debug_info[VALUES] = values_score

    # Name Logic
    if prediction_factors_weights.get(NAME, 0) > 0:
        if not metadata.name or not metadata.name.strip():
            debug_info[NAME] = f"0.0 (Blank {NAME} Metadata)"
        else:
            debug_info[NAME] = match_regex(metadata.name, config[NAME][REGEX])

    # Description_Logic
    if prediction_factors_weights.get(DESCRIPTION, 0) > 0:
        if not metadata.description or not metadata.description.strip():
            debug_info[DESCRIPTION] = f"0.0 (Blank {DESCRIPTION} Metadata)"
        else:
            debug_info[DESCRIPTION] = match_regex(
                metadata.description, config[DESCRIPTION][REGEX]
            )

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE] = match_datatype(
                metadata.datatype, config[DATATYPE][TYPE]
            )

    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key]) != str:
            confidence_level += prediction_factors_weights[key] * debug_info[key]
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_iban(metadata, values, config):  # noqa: C901
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info: Dict[str, Any] = {}

    # Value Logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0.0
        try:
            if config[VALUES][PREDICTION_TYPE] == "regex":
                values_score = match_regex_for_values(values, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == "library":
                iban_score = 0.0
                for val in values:
                    try:
                        if IBAN(val, allow_invalid=True).is_valid:
                            iban_score += 1.0
                    except Exception:
                        pass
                values_score = iban_score / len(values)
            else:
                raise Exception(
                    f"Inappropriate values_prediction_type {config[VALUES][PREDICTION_TYPE]}"
                )
        except Exception as e:
            logger.error(f"Column {metadata.name} failed due to {e}")
        values_score = np.round(values_score, 2)
        debug_info[VALUES] = values_score

    # Name Logic
    if prediction_factors_weights.get(NAME, 0) > 0:
        if not metadata.name or not metadata.name.strip():
            debug_info[NAME] = f"0.0 (Blank {NAME} Metadata)"
        else:
            debug_info[NAME] = match_regex(metadata.name, config[NAME][REGEX])

    # Description_Logic
    if prediction_factors_weights.get(DESCRIPTION, 0) > 0:
        if not metadata.description or not metadata.description.strip():
            debug_info[DESCRIPTION] = f"0.0 (Blank {DESCRIPTION} Metadata)"
        else:
            debug_info[DESCRIPTION] = match_regex(
                metadata.description, config[DESCRIPTION][REGEX]
            )

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE] = match_datatype(
                metadata.datatype, config[DATATYPE][TYPE]
            )

    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key]) != str:
            confidence_level += prediction_factors_weights[key] * debug_info[key]
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_vehicle_identification_number(metadata, values, config):  # noqa: C901
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info: Dict[str, Any] = {}

    # Value Logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0.0
        try:
            if config[VALUES][PREDICTION_TYPE] == "regex":
                values_score = match_regex_for_values(values, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == "library":
                vin_score = 0
                for val in values:
                    try:
                        # Vin constructor implicitly validates the VIN
                        _ = Vin(val)
                        vin_score += 1
                    except Exception:
                        pass
                values_score = vin_score / len(values)
            else:
                raise Exception(
                    f"Inappropriate values_prediction_type {config[VALUES][PREDICTION_TYPE]}"
                )
        except Exception as e:
            logger.error(f"Column {metadata.name} failed due to {e}")
        values_score = np.round(values_score, 2)
        debug_info[VALUES] = values_score

    # Name Logic
    if prediction_factors_weights.get(NAME, 0) > 0:
        if not metadata.name or not metadata.name.strip():
            debug_info[NAME] = f"0.0 (Blank {NAME} Metadata)"
        else:
            debug_info[NAME] = match_regex(metadata.name, config[NAME][REGEX])

    # Description_Logic
    if prediction_factors_weights.get(DESCRIPTION, 0) > 0:
        if not metadata.description or not metadata.description.strip():
            debug_info[DESCRIPTION] = f"0.0 (Blank {DESCRIPTION} Metadata)"
        else:
            debug_info[DESCRIPTION] = match_regex(
                metadata.description, config[DESCRIPTION][REGEX]
            )

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE] = match_datatype(
                metadata.datatype, config[DATATYPE][TYPE]
            )

    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key]) != str:
            confidence_level += prediction_factors_weights[key] * debug_info[key]
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_ip_address_v4(metadata, values, config):  # noqa: C901
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info: Dict[str, Any] = {}

    # Value Logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0.0
        try:
            if config[VALUES][PREDICTION_TYPE] == "regex":
                values_score = match_regex_for_values(values, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == "library":
                count = 0
                for value in values:
                    try:
                        _ = ipaddress.IPv4Address(str(value))
                        count = count + 1
                    except Exception:
                        pass
                values_score = count / len(values)
            else:
                raise Exception(
                    f"Inappropriate values_prediction_type {config[VALUES][PREDICTION_TYPE]}"
                )
        except Exception as e:
            logger.error(f"Column {metadata.name} failed due to {e}")
        values_score = np.round(values_score, 2)
        debug_info[VALUES] = values_score

    # Name Logic
    if prediction_factors_weights.get(NAME, 0) > 0:
        if not metadata.name or not metadata.name.strip():
            debug_info[NAME] = f"0.0 (Blank {NAME} Metadata)"
        else:
            debug_info[NAME] = match_regex(metadata.name, config[NAME][REGEX])

    # Description_Logic
    if prediction_factors_weights.get(DESCRIPTION, 0) > 0:
        if not metadata.description or not metadata.description.strip():
            debug_info[DESCRIPTION] = f"0.0 (Blank {DESCRIPTION} Metadata)"
        else:
            debug_info[DESCRIPTION] = match_regex(
                metadata.description, config[DESCRIPTION][REGEX]
            )

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE] = match_datatype(
                metadata.datatype, config[DATATYPE][TYPE]
            )

    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key]) != str:
            confidence_level += prediction_factors_weights[key] * debug_info[key]
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_ip_address_v6(metadata, values, config):  # noqa: C901
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info: Dict[str, Any] = {}

    # Value Logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0.0
        try:
            if config[VALUES][PREDICTION_TYPE] == "regex":
                values_score = match_regex_for_values(values, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == "library":
                count = 0
                for value in values:
                    try:
                        _ = ipaddress.IPv6Address(str(value))
                        count = count + 1
                    except Exception:
                        pass
                values_score = count / len(values)
            else:
                raise Exception(
                    f"Inappropriate values_prediction_type {config[VALUES][PREDICTION_TYPE]}"
                )
        except Exception as e:
            logger.error(f"Column {metadata.name} failed due to {e}")
        values_score = np.round(values_score, 2)
        debug_info[VALUES] = values_score

    # Name Logic
    if prediction_factors_weights.get(NAME, 0) > 0:
        if not metadata.name or not metadata.name.strip():
            debug_info[NAME] = f"0.0 (Blank {NAME} Metadata)"
        else:
            debug_info[NAME] = match_regex(metadata.name, config[NAME][REGEX])

    # Description_Logic
    if prediction_factors_weights.get(DESCRIPTION, 0) > 0:
        if not metadata.description or not metadata.description.strip():
            debug_info[DESCRIPTION] = f"0.0 (Blank {DESCRIPTION} Metadata)"
        else:
            debug_info[DESCRIPTION] = match_regex(
                metadata.description, config[DESCRIPTION][REGEX]
            )

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE] = match_datatype(
                metadata.datatype, config[DATATYPE][TYPE]
            )

    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key]) != str:
            confidence_level += prediction_factors_weights[key] * debug_info[key]
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_us_driving_license_number(metadata, values, config):
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info: Dict[str, Any] = {}

    # Value Logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0.0
        try:
            if config[VALUES][PREDICTION_TYPE] == "regex":
                values_score = match_regex_for_values(values, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == "library":
                raise Exception(
                    "Currently prediction type 'library' is not supported for "
                    "infotype US Driving License Number"
                )
            else:
                raise Exception(
                    f"Inappropriate values_prediction_type {config[VALUES][PREDICTION_TYPE]}"
                )
        except Exception as e:
            logger.error(f"Column {metadata.name} failed due to {e}")
        values_score = np.round(values_score, 2)
        debug_info[VALUES] = values_score

    # Name Logic
    if prediction_factors_weights.get(NAME, 0) > 0:
        if not metadata.name or not metadata.name.strip():
            debug_info[NAME] = f"0.0 (Blank {NAME} Metadata)"
        else:
            debug_info[NAME] = match_regex(metadata.name, config[NAME][REGEX])

    # Description_Logic
    if prediction_factors_weights.get(DESCRIPTION, 0) > 0:
        if not metadata.description or not metadata.description.strip():
            debug_info[DESCRIPTION] = f"0.0 (Blank {DESCRIPTION} Metadata)"
        else:
            debug_info[DESCRIPTION] = match_regex(
                metadata.description, config[DESCRIPTION][REGEX]
            )

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE] = match_datatype(
                metadata.datatype, config[DATATYPE][TYPE]
            )

    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key]) != str:
            confidence_level += prediction_factors_weights[key] * debug_info[key]
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_us_social_security_number(metadata, values, config):  # noqa: C901
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info: Dict[str, Any] = {}

    # Value Logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0.0
        try:
            if config[VALUES][PREDICTION_TYPE] == "regex":
                values_score = match_regex_for_values(values, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == "library":
                us_ssn_score = 0
                for val in values:
                    try:
                        if us_ssn.is_valid(val):
                            us_ssn_score += 1
                    except Exception:
                        pass
                values_score = us_ssn_score / len(values)
            else:
                raise Exception(
                    f"Inappropriate values_prediction_type {config[VALUES][PREDICTION_TYPE]}"
                )
        except Exception as e:
            logger.error(f"Column {metadata.name} failed due to {e}")
        values_score = np.round(values_score, 2)
        debug_info[VALUES] = values_score

    # Name Logic
    if prediction_factors_weights.get(NAME, 0) > 0:
        if not metadata.name or not metadata.name.strip():
            debug_info[NAME] = f"0.0 (Blank {NAME} Metadata)"
        else:
            debug_info[NAME] = match_regex(metadata.name, config[NAME][REGEX])

    # Description_Logic
    if prediction_factors_weights.get(DESCRIPTION, 0) > 0:
        if not metadata.description or not metadata.description.strip():
            debug_info[DESCRIPTION] = f"0.0 (Blank {DESCRIPTION} Metadata)"
        else:
            debug_info[DESCRIPTION] = match_regex(
                metadata.description, config[DESCRIPTION][REGEX]
            )

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE] = match_datatype(
                metadata.datatype, config[DATATYPE][TYPE]
            )

    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key]) != str:
            confidence_level += prediction_factors_weights[key] * debug_info[key]
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_swift_code(metadata, values, config):  # noqa: C901
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info: Dict[str, Any] = {}

    # Value Logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0.0
        try:
            if config[VALUES][PREDICTION_TYPE] == "regex":
                values_score = match_regex_for_values(values, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == "library":
                swift_score = 0
                for val in values:
                    try:
                        if BIC(val, allow_invalid=True).is_valid:
                            swift_score += 1
                    except Exception:
                        pass
                values_score = swift_score / len(values)
            else:
                raise Exception(
                    f"Inappropriate values_prediction_type {config[VALUES][PREDICTION_TYPE]}"
                )
        except Exception as e:
            logger.error(f"Column {metadata.name} failed due to {e}")
        values_score = np.round(values_score, 2)
        debug_info[VALUES] = values_score

    # Name Logic
    if prediction_factors_weights.get(NAME, 0) > 0:
        if not metadata.name or not metadata.name.strip():
            debug_info[NAME] = f"0.0 (Blank {NAME} Metadata)"
        else:
            debug_info[NAME] = match_regex(metadata.name, config[NAME][REGEX])

    # Description_Logic
    if prediction_factors_weights.get(DESCRIPTION, 0) > 0:
        if not metadata.description or not metadata.description.strip():
            debug_info[DESCRIPTION] = f"0.0 (Blank {DESCRIPTION} Metadata)"
        else:
            debug_info[DESCRIPTION] = match_regex(
                metadata.description, config[DESCRIPTION][REGEX]
            )

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE] = match_datatype(
                metadata.datatype, config[DATATYPE][TYPE]
            )
    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key]) != str:
            confidence_level += prediction_factors_weights[key] * debug_info[key]
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info
