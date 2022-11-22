
from fuzzywuzzy import fuzz
## GLOVE URL :  https://nlp.stanford.edu/data/glove.6B.zip
from nltk.corpus import stopwords
from numpy.linalg import norm
import re
import numpy as np
from helper_classes import TableInfo, ColumnInfo
import logging
import re
# from datahub_classify.constants import PREDICTION_FACTORS_AND_WEIGHTS, VALUES

logger = logging.getLogger(__name__)
glove_vec = "C:\\Users\\GS-3490\\Glossary_creation\\Glossary_stage_2\\glove.6B\\glove.6B.50d.txt"
stop_words = set(stopwords.words('english'))

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


column_desc_threshold = 0.65
column_weighted_score_threshold = 0.7
column_lineage_threshold = 0.7
table_similarity_threshold = 0.7

table_desc_threshold = 0.65
table_weighted_score_threshold = 0.7



##### Stage -1 Utils

# TODO: Exception handling
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
            # TODO: print the exception for debugging purpose
            logger.error(f"Regex match for values failed due to: {e}", exc_info=e)
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



### Stage 2 Utils

def cosine_similarity_score(vec1,
                            vec2):
    cos_sim = np.dot(vec1, vec2)/ (norm(vec1) * norm(vec2))
    if cos_sim <=0:
        cos_sim = 0
    # return  (1 + cos_sim) / 2
    return cos_sim

def read_glove_vector(glove_vec: str):
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
    return word_to_vec_map


word_to_vec_map = read_glove_vector(glove_vec)


# def name_desc_similarity(text_1: str,
#                          text_2: str):
#     text_1 = text_1.strip()
#     text_2 = text_2.strip()

#     if (text_1 not in ([None,""])) and (text_2 not in ([None,""])):
#         fuzzy_match_score = fuzz.partial_ratio(text_1, text_2) / 100
#         if fuzzy_match_score <= 0.5:
#             fuzzy_match_score = 0.8 * fuzzy_match_score
#         text_1_cleaned = re.sub(r"[^a-z]", " ", text_1.lower())
#         text_2_cleaned = re.sub(r"[^a-z]", " ", text_2.lower())
#         text_1_words= [word for word in text_1_cleaned.split() if word not in (stop_words)]
#         text_2_words = [word for word in text_2_cleaned.split() if word not in (stop_words)]
#         text_1_avg_glove_emb = np.array(
#             [word_to_vec_map[word] for word in text_1_words if word in word_to_vec_map.keys()]).mean(axis=0)
#         text_2_avg_glove_emb = np.array(
#             [word_to_vec_map[word] for word in text_2_words if word in word_to_vec_map.keys()]).mean(axis=0)

#         # print(text_1_avg_glove_emb)
#         emb_match_score = None
#         if text_1_avg_glove_emb.shape[0] >0 and text_2_avg_glove_emb.shape[0] >0:
#             emb_match_score = cosine_similarity_score(text_1_avg_glove_emb, text_2_avg_glove_emb)

#         if emb_match_score:
#             score = np.maximum(fuzzy_match_score, emb_match_score)
#         else:
#             score = fuzzy_match_score
#     else:
#         score=0

#     print(text_1)
#     print(text_2)
#     print("fuzzy score: ", fuzzy_match_score)
#     print("glove score: ", emb_match_score)
#     return score



def name_desc_similarity(text_1: str,
                         text_2: str):
    text_1 = text_1.lower().strip()
    text_2 = text_2.lower().strip()

    if (text_1 not in ([None,""])) and (text_2 not in ([None,""])):
        fuzzy_match_score = fuzz.token_set_ratio(text_1, text_2) / 100
        if fuzzy_match_score <= 0.5:
            fuzzy_match_score = 0.8 * fuzzy_match_score

        text_1_cleaned = re.sub(r"[^a-z]", " ", text_1.lower()).strip()
        text_2_cleaned = re.sub(r"[^a-z]", " ", text_2.lower()).strip()
        text_1_words= [word for word in text_1_cleaned.split() if word not in (stop_words)]
        text_2_words = [word for word in text_2_cleaned.split() if word not in (stop_words)]

        if len(text_1_words)<=1 and len(text_2_words)<=1:
            emb_1 = np.array(
            [word_to_vec_map[word] for word in text_1_words if word in word_to_vec_map.keys()]).mean(axis=0)
            emb_2 = np.array(
            [word_to_vec_map[word] for word in text_2_words if word in word_to_vec_map.keys()]).mean(axis=0)

            if str(emb_1) == "nan" or str(emb_2) == "nan":
                embeddings = model.encode([text_1, text_2])
                emb_1 = embeddings[0]
                emb_2 = embeddings[1]

        else:
            embeddings = model.encode([text_1, text_2])
            emb_1 = embeddings[0]
            emb_2 = embeddings[1]

        emb_match_score = cosine_similarity_score(emb_1, emb_2)
        score =  np.maximum(fuzzy_match_score, emb_match_score)

    else:
        score=0

    # print(text_1)
    # print(text_2)
    # print("fuzzy score: ", fuzzy_match_score)
    # print("glove score: ", emb_match_score)
    return score


def column_dtype_similarity(column_1_dtype: str,
                            column_2_dtype: str):
    if column_1_dtype == column_2_dtype:
        column_dtype_score = 1
    else:
        column_dtype_score = 0
    return column_dtype_score


def table_platform_similarity(platform_1_name: str,
                              platform_2_name: str):
    if not platform_1_name == platform_2_name:
        platform_score = 1
    else:
        platform_score = 0
    return platform_score

def compute_lineage_score(entity_1_parents: list,
                          entity_2_parents: list,
                          entity_1_id: str,
                          entity_2_id: str):
    if (entity_1_id in entity_2_parents) or (entity_2_id in entity_1_parents):
        lineage_score = 1
    else:
        lineage_score= 0
    return lineage_score

def table_schema_similarity(table_1_cols_name_dtypes: list[tuple],
                            table_2_cols_names_dtypes: list[tuple],
                            pair_score_threshold: float = 0.7):
    matched_pairs = []
    num_cols_table1 = len(table_1_cols_name_dtypes)
    num_cols_table2 = len(table_2_cols_names_dtypes)
    for col1 in table_1_cols_name_dtypes:
        for col2 in table_2_cols_names_dtypes:
            col1_name = col1[0]
            col2_name = col2[0]
            col1_dtype = col1[1]
            col2_dtype =col2[1]
            col_name_score = name_desc_similarity(col1_name, col2_name)
            col_dtype_score = column_dtype_similarity(col1_dtype, col2_dtype)
            pair_score = 0.7*col_name_score + 0.3*col_dtype_score

            if pair_score > pair_score_threshold:
                matched_pairs.append((col1, col2))
                table_2_cols_names_dtypes.remove(col2)
                break
    schema_score = 2* len(matched_pairs) / (num_cols_table1 + num_cols_table2)
    return schema_score


def compute_table_overall_similarity_score(name_score: float,
                                           desc_score: float,
                                           platform_score: bool,
                                           lineage_score: bool,
                                           schema_score: float,
                                           desc_present: bool):
    weighted_score = 0.3*name_score + 0.1*platform_score +  0.6*schema_score
    if desc_present:
        if desc_score >= table_desc_threshold:
            weighted_score = np.minimum(1.1 * weighted_score, 1)
        else:
            weighted_score = 0.95*weighted_score
    
    overall_table_similarity_score = weighted_score

    if weighted_score >= table_weighted_score_threshold:
        if lineage_score ==1:
            overall_table_similarity_score = 1
   
    return np.round(overall_table_similarity_score,2)

def compute_column_overall_similarity_score(name_score: float,
                                            dtype_score: bool,
                                            desc_score: float,
                                            table_similarity_score: float,
                                            lineage_score: bool,
                                            desc_present: bool):
    weighted_score = 0.8* name_score + 0.2* dtype_score
    if desc_present:
        if desc_score >= column_desc_threshold:
            weighted_score = 1.1 * weighted_score
        else:
            weighted_score = 0.95*weighted_score

    if weighted_score > column_weighted_score_threshold:
        if table_similarity_score > table_similarity_threshold:
            weighted_score = 1.05 * weighted_score
        if lineage_score ==1:
            weighted_score = 1.1* weighted_score

    overall_column_similarity_score = np.minimum(weighted_score, 1)
    
  
    return np.round(overall_column_similarity_score, 2)


def compute_table_similarity(table_info1: TableInfo,
                             table_info2: TableInfo):
    table1_name = table_info1.metadata.name
    table1_desc = table_info1.metadata.description
    table1_platform = table_info1.metadata.platform
    table1_parents = table_info1.parent_tables
    table_1_cols_name_dtypes = list()
    for col_info in table_info1.column_infos:
        table_1_cols_name_dtypes.append((col_info.metadata.name, col_info.metadata.datatype))

    table2_name = table_info2.metadata.name
    table2_desc = table_info2.metadata.description
    table2_platform =table_info2.metadata.platform
    table2_parents = table_info2.parent_tables
    table_2_cols_name_dtypes = list()
    for col_info in table_info2.column_infos:
        table_2_cols_name_dtypes.append((col_info.metadata.name, col_info.metadata.datatype))


    if table1_desc and table2_desc:
        desc_present=True
    else:
        desc_present = False

    table_name_score = name_desc_similarity(table1_name, table2_name)
    table_desc_score = name_desc_similarity(table1_desc, table2_desc)
    table_platform_score = table_platform_similarity (table1_platform, table2_platform )
    table_lineage_score = compute_lineage_score(table1_parents, table2_parents, table1_name, table2_name)
    table_schema_score = table_schema_similarity(table_1_cols_name_dtypes, table_2_cols_name_dtypes)

    print("name score: ", table_name_score)
    print("desc score: " ,table_desc_score)
    print("platforms: ", table1_platform, table2_platform)
    print("platform score: ", table_platform_score)
    print("lineae score: ", table_lineage_score)
    print("schema score: ", table_schema_score)
    print("======================================")

    overall_table_similarity_score = compute_table_overall_similarity_score(table_name_score,
                                                                            table_desc_score,
                                                                            table_platform_score,
                                                                            table_lineage_score,
                                                                            table_schema_score,
                                                                            desc_present)
    return overall_table_similarity_score


def compute_column_similarity(col_info1: ColumnInfo,
                              col_info2: ColumnInfo,
                              overall_table_similarity_score: float):
    column1_name = col_info1.metadata.name
    column1_desc = col_info1.metadata.description
    column1_dtype = col_info1.metadata.datatype
    column1_parents = col_info1.parent_columns

    column2_name = col_info2.metadata.name
    column2_desc = col_info2.metadata.description
    column2_dtype = col_info2.metadata.datatype
    column2_parents = col_info2.parent_columns

    if column1_desc and column2_desc:
        desc_present=True
    else:
        desc_present = False

    column_name_score = name_desc_similarity(column1_name, column2_name)
    if desc_present:
        column_desc_score = name_desc_similarity(column1_desc, column2_desc)
    else:
        column_desc_score = None
    column_dtype_score = column_dtype_similarity(column1_dtype, column2_dtype)
    column_lineage_score = compute_lineage_score(column1_parents, column2_parents, column1_name, column2_name)
    overall_column_similarity_score = compute_column_overall_similarity_score(column_name_score,
                                                                              column_dtype_score,
                                                                              column_desc_score,
                                                                              overall_table_similarity_score,
                                                                              column_lineage_score,
                                                                              desc_present)

    print("pair: ", (col_info1.metadata.column_id, col_info2.metadata.column_id ))
    print("name score: ", column_name_score)
    print("-------------")
    print(column1_desc)
    print(column2_desc)
    print("-------------")
    print("desc score: " ,column_desc_score)
    print("dtype score: ", column_dtype_score)
    print("lineage score: ", column_lineage_score)
    print("overall score: ", overall_column_similarity_score)
    print("****************************")
    # print("schema score: ", table_schema_score)
    return overall_column_similarity_score

