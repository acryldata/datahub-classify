import re


# TODO: Exception handling
# Match regex for Name and Description
def match_regex(text_to_match, regex_list):
    original_text = text_to_match.lower()
    cleaned_text = ''.join(e for e in original_text if e.isalpha())
    match_score = 0
    for pattern in regex_list:
        try:
            cleaned_pattern = ''.join(e for e in pattern if e.isalpha())
            if (cleaned_pattern == cleaned_text) or (re.fullmatch(pattern, original_text)):
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
            pass
    match_score = round(match_score, 2)
    return match_score


# Match data type
def match_datatype(dtype_to_match, dtype_list):
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
        except:
            pass
    values_score = sum(values_score_list) / length_values
    return values_score
