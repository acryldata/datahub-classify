import pandas as pd
import numpy as np
import spacy
import traceback
import phonenumbers
import re

from infotype_utils import match_regex, match_datatype, match_regex_for_values, detect_named_entity_spacy
from constants import *

nlp_english = spacy.load('en_core_web_sm')
spacy_models_list = [nlp_english]


def inspect_for_email_address(metadata, values, config):
    # TODO: change the variable name "prediction_factors_weights" to "prediction_factors_weights"
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info = {}

    # Value Logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0
        if config[VALUES][PREDICTION_TYPE] == 'regex':
            values_score = match_regex_for_values(values, config[VALUES][REGEX])
        elif config[VALUES][PREDICTION_TYPE] == 'library':
            raise "Currently prediction type 'library' is not supported for infotype Email Address"
        else:
            raise "Inappropriate Prediction type %s" % config[VALUES][PREDICTION_TYPE]
        values_score = np.round(values_score, 2)
        debug_info[VALUES] = values_score


    # TODO: Keep only debug_info dict,
    # TODO: probably we don't require two separate dictionaries (blank_metadata & metadata_score)
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
            debug_info[DESCRIPTION] = match_regex(metadata.description, config[DESCRIPTION][REGEX])

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE]  = match_datatype(metadata.datatype, config[DATATYPE][TYPE])

    # TODO: modify the code considering only one dictionary i.e. debug_info{}
    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key])!= str:
            confidence_level += np.round(prediction_factors_weights[key] * debug_info[key], 2)
    confidence_level = np.round(confidence_level, 2)
    return confidence_level, debug_info


def inspect_for_street_address(metadata, values, config):
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info = {}

    # Values logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0
        try:
            if config[VALUES][PREDICTION_TYPE] == 'regex':
                raise "Currently prediction type 'regex' is not supported for infotype Street Address"
            elif config[VALUES][PREDICTION_TYPE] == 'library':
                entity_count = 0
                entities_of_interest = ["FAC", "LOC", "ORG"]
                weight = 1.5
                for value in values:
                    # TODO: think of spacy common code into a function,
                    # TODO: use this common function at in other functions (inspect_for_full_name)
                    # TODO: rename nlp_list to spacy_model_list, also think of renaming 'nlp' to some intuitive name
                    if detect_named_entity_spacy(spacy_models_list, entities_of_interest, value):
                        entity_count += 1*weight
                entities_score = entity_count / len(values)
                values_score = np.minimum(entities_score, 1)
            else:
                raise "Inappropriate values_prediction_type %s" % config[VALUES][PREDICTION_TYPE]
        except Exception as e:
            # traceback.print_exc()
            # values_score = 0
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
            debug_info[DESCRIPTION] = match_regex(metadata.description, config[DESCRIPTION][REGEX])

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE]  = match_datatype(metadata.datatype, config[DATATYPE][TYPE])

    # TODO: modify the code considering only one dictionary i.e. debug_info{}
    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key])!= str:
            confidence_level += np.round(prediction_factors_weights[key] * debug_info[key], 2)
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_gender(metadata, values, config):
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info = {}

    # Value Logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0
        try:
            values = pd.Series(values).astype(str)
            if config[VALUES][PREDICTION_TYPE] == 'regex':
                values_score = match_regex_for_values(values, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == 'library':
                raise "Currently prediction type 'library' is not supported for infotype Gender"
            else:
                raise "Inappropriate values_prediction_type %s" % config[VALUES][PREDICTION_TYPE]
            if values_score == 0:
                if values.nunique() <= 5:  # TODO: check possible appropriate unique values of gender
                    values_score = 0.3  # TODO: instead of 0.5 let's use weight as 0.3
        except Exception as e:
            # traceback.print_exc()
            # values_score = 0
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
            debug_info[DESCRIPTION] = match_regex(metadata.description, config[DESCRIPTION][REGEX])

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE]  = match_datatype(metadata.datatype, config[DATATYPE][TYPE])

    # TODO: modify the code considering only one dictionary i.e. debug_info{}
    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key])!= str:
            confidence_level += np.round(prediction_factors_weights[key] * debug_info[key], 2)
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_credit_card_number(metadata, values, config):
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info = {}

    # Value Logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0
        try:
            values = pd.Series(values).astype(str)
            values_cleaned = []
            for value in values:
                string_cleaned = re.sub(r"[^a-zA-Z0-9]+", "", value)  # TODO: do we require this cleaning step?
                values_cleaned.append(string_cleaned)
            if config[VALUES][PREDICTION_TYPE] == 'regex':
                values_score = match_regex_for_values(values_cleaned, config[VALUES][REGEX])
            elif config[VALUES][PREDICTION_TYPE] == 'library':
                raise "Currently prediction type 'library' is not supported for infotype Credit Card Number"
            else:
                raise "Inappropriate values_prediction_type %s" % config[VALUES][PREDICTION_TYPE]
        except Exception as e:
            # traceback.print_exc()
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
            debug_info[DESCRIPTION] = match_regex(metadata.description, config[DESCRIPTION][REGEX])

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE]  = match_datatype(metadata.datatype, config[DATATYPE][TYPE])

    # TODO: modify the code considering only one dictionary i.e. debug_info{}
    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key])!= str:
            confidence_level += np.round(prediction_factors_weights[key] * debug_info[key], 2)
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_phone_number(metadata, values, config):
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info = {}

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

    # Values logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0
        try:
            if config[VALUES][PREDICTION_TYPE] == 'regex':
                raise "Currently prediction type 'regex' is not supported for infotype Phone Number"
            elif config[VALUES][PREDICTION_TYPE] == 'library':
                valid_phone_numbers_count = 0
                for value in values:
                    for code in iso_codes:
                        parsed_number = phonenumbers.parse(value, code)
                        if phonenumbers.is_possible_number(parsed_number):
                            valid_phone_numbers_count += 1
                            break
                values_score = valid_phone_numbers_count / len(values)
            else:
                raise "Inappropriate values_prediction_type %s" % config[VALUES][PREDICTION_TYPE]
        except Exception as e:
            # traceback.print_exc()
            # values_score = 0
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
            debug_info[DESCRIPTION] = match_regex(metadata.description, config[DESCRIPTION][REGEX])

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE]  = match_datatype(metadata.datatype, config[DATATYPE][TYPE])

    # TODO: modify the code considering only one dictionary i.e. debug_info{}
    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key])!= str:
            confidence_level += np.round(prediction_factors_weights[key] * debug_info[key], 2)
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_full_name(metadata, values, config):
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info = {}

    # Values logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0
        try:
            if config[VALUES][PREDICTION_TYPE] == 'regex':
                raise "Currently prediction type 'regex' is not supported for infotype Phone Number"
            elif config[VALUES][PREDICTION_TYPE] == 'library':
                entity_count = 0
                entities_of_interest = ["PERSON"]
                weight = 1
                for value in values:
                    # TODO: think of spacy common code into a function,
                    # TODO: use this common function at in other functions (inspect_for_full_name)
                    # TODO: rename nlp_list to spacy_model_list, also think of renaming 'nlp' to some intuitive name
                    if detect_named_entity_spacy(spacy_models_list, entities_of_interest, value):
                        entity_count += 1 * weight
                entities_score = entity_count / len(values)
                values_score = np.minimum(entities_score, 1)
            else:
                raise "Inappropriate values_prediction_type %s" % config[VALUES][PREDICTION_TYPE]
        except Exception as e:
            # traceback.print_exc()
            # values_score = 0
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
            debug_info[DESCRIPTION] = match_regex(metadata.description, config[DESCRIPTION][REGEX])

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE]  = match_datatype(metadata.datatype, config[DATATYPE][TYPE])

    # TODO: modify the code considering only one dictionary i.e. debug_info{}
    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key])!= str:
            confidence_level += np.round(prediction_factors_weights[key] * debug_info[key], 2)
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info


def inspect_for_age(metadata, values, config):
    prediction_factors_weights = config[PREDICTION_FACTORS_AND_WEIGHTS]
    debug_info = {}

    # Values logic
    if prediction_factors_weights.get(VALUES, 0) > 0:
        values_score = 0
        try:
            if config[VALUES][PREDICTION_TYPE] == 'regex':
                raise "Currently prediction type 'regex' is not supported for infotype Phone Number"
            elif config[VALUES][PREDICTION_TYPE] == 'library':
                values_series = pd.Series(values).dropna()
                # Check if column is convertible to int dtype
                int_col = values_series.astype(int)
                # Check is fraction values are present
                bool_out = np.round(values_series) == values_series
                if bool_out.all():
                    max_val = int_col.max()
                    min_val = int_col.min()
                    age_range = max_val - min_val
                    num_unique = int_col.nunique()
                    if max_val <= 120 and min_val >= 0:
                        # Add 0.7 score if all values are within [0, 120]
                        values_score += 0.7
                        # Add 0.1 score if range is more than np.minimum(len(df)/50, 60)
                        if age_range > np.minimum(len(values) / 50, 60):
                            values_score += 0.1
                        # Add 0.2 score if num unique values is more than np.minimum(len(df)/50, 40)
                        if num_unique >= np.minimum(len(values) / 50, 40):
                            values_score += 0.2
                    else:
                        values_score = 0
                else:
                    values_score = 0
            else:
                raise "Inappropriate values_prediction_type %s" % config[VALUES][PREDICTION_TYPE]
        except Exception as e:
            # traceback.print_exc()
            # values_score = 0
            pass

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
            debug_info[DESCRIPTION] = match_regex(metadata.description, config[DESCRIPTION][REGEX])

    # Datatype_Logic
    if prediction_factors_weights.get(DATATYPE, 0) > 0:
        if not metadata.datatype or not metadata.datatype.strip():
            debug_info[DATATYPE] = f"0.0 (Blank {DATATYPE} Metadata)"
        else:
            debug_info[DATATYPE]  = match_datatype(metadata.datatype, config[DATATYPE][TYPE])

    # TODO: modify the code considering only one dictionary i.e. debug_info{}
    confidence_level = 0
    for key in debug_info.keys():
        if type(debug_info[key])!= str:
            confidence_level += np.round(prediction_factors_weights[key] * debug_info[key], 2)
    confidence_level = np.round(confidence_level, 2)

    return confidence_level, debug_info
