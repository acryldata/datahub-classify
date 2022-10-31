# datahub-classify
Predict InfoTypes for DataHub
# “predict_infotypes” API
This API populates infotype proposal(s) for each input column by using metadata, values & confidence level threshold. Following are the input and output contract
### Input Contract:
API expects following parameters in the output
- _column_infos_ - This is a list of ColumnInfo objects. Each ColumnInfo object contains metadata (col_name, description, datatype, etc) and values of a column. 
- _confidence_level_threshold_ - If the infotype prediction confidence is greater than the confidence threshold then the prediction is considered as a proposal.
- _global_config_ - This dictionary contains configuration details about all supported infotypes. Refer section X for more information.
### Output Contract:
API returns a list of ColumnInfo objects of length same as input ColumnInfo objects list. A populated list of Infotype proposal(s), if any, is added in the ColumnInfo object itself with a variable name as infotype_proposals. The infotype_proposals list contains InfotypeProposal objects which has following information
- _infotype_ - A proposed infotype name.
- _confidence_level_ - Overall confidence of the infotype proposal. 
- _debug _info_ - confidence score of each prediction factor involved in the overall confidence score calculation. Refer section X for more information.

**Convention:**
If infotype_proposals list is non-empty then it indicates that there is at least one infotype proposal with confidence greater than confidence_level_threshold.
# Infotype Configuration
Infotype configuration is a dictionary with all infotypes at root level key. Each infotype has following configurable parameters (value of each parameter is a dictionary)
- _Prediction_Factors_and_Weights_ - This is a dictionary that specifies the weight of each prediction factor which will be used in the final confidence calculation. Following are the prediction factors
  1. Name 
  2. Description 
  3. Datatype 
  4. Values
- _Name_ - regex list which is to be matched against column name
- _Description_ - regex list which is to be matched against column description
- _Datatype_ - list of datatypes to be matched against column datatype
- _Values_ - this dictionary contains following information
  1. _prediction_type_ - values evaluation model (regex/library)
  2. _regex_ - regex list which is to be matched against column values
  3. _library_ - library name which is to be used to evaluate column values

### Sample Infotype Configuration Dictionary
    {
        '<Infotype1>': {
            'Prediction_Factors_and_Weights': {
                'Name': 0.4,
                'Description': 0,
                'Datatype': 0,
                'Values': 0.6
            },
            'Name': { 'regex': [<regex patterns>] },
            'Description': { 'regex': [<regex patterns>] },
            'Datatype': { 'type': [<list of datatypes>] },
            'Values': {
                'prediction_type': 'regex/library',
                'regex': [<regex patterns>],
                'library': [<library name>]
            }
        },
        '<Infotype2>': {
        ..
        ..
        ..
        }
    }
# Debug Information
A debug information is associated with each infotype proposal, it provides details about confidence score from each prediction factor involved in overall confidence score calculation. This is a dictionary with following four prediction factors as key
- Name 
- Description 
- Datatype 
- Values
# Supported Infotypes
1. Age 
2. Gender 
3. Person Name / Full Name 
4. Email Address 
5. Phone Number 
6. Street Address 
7. Credit-Debit Card Number 
# Required Libraries
Following libraries are required
- Spacy 3.4.1 
- phonenumbers 8.12.56
#### Required Spacy model
$ python3 -m spacy download en_core_web_sm
# Assumptions
- If value prediction factor weight is non zero (indicating values should be used for infotype inspection) then a minimum 50 non-null column values should be present.
# How to Run
    $ cd <datahub-classify repo root folder>
    $ python3 -m venv venv
    $ source venv/bin/activate
    $ pip install -e .
    $ cd test
    $ python3 -m pytest sample_testing.py --capture=no -s --log-cli-level=DEBUG
