# datahub-classify

Predict InfoTypes for [DataHub](https://datahubproject.io/).

## Installation

`python3 -m pip install --upgrade acryl-datahub-classify`

# API 'predict_infotypes'

This API populates infotype proposal(s) for each input column by using metadata, values & confidence level threshold. Following are the input and output contract

### API Input

API expects following parameters in the output

- `column_infos` - This is a list of ColumnInfo objects. Each ColumnInfo object contains metadata (col_name, description, datatype, etc) and values of a column.
- `confidence_level_threshold` - If the infotype prediction confidence is greater than the confidence threshold then the prediction is considered as a proposal. This is the common threshold for all infotypes.
- `global_config` - This dictionary contains configuration details about all supported infotypes. Refer section [Infotype Configuration](#infotype-configuration) for more information.
- `infotypes` - This is a list of infotypes that is to be processed. This is an optional argument, if specified then it will override the default list of all supported infotypes. If user is interested in only few infotypes then this list can be specified with correct infotype names. Infotype names are case sensitive.

### API Output

API returns a list of ColumnInfo objects of length same as input ColumnInfo objects list. A populated list of Infotype proposal(s), if any, is added in the ColumnInfo object itself with a variable name as `infotype_proposals`. The infotype_proposals list contains InfotypeProposal objects which has following information

- `infotype` - A proposed infotype name.
- `confidence_level` - Overall confidence of the infotype proposal.
- `debug_info` - confidence score of each prediction factor involved in the overall confidence score calculation. Refer section [Debug Information](#debug-information) for more information.

**Convention:**
If `infotype_proposals` list is non-empty then it indicates that there is at least one infotype proposal with confidence greater than `confidence_level_threshold`.

### Infotype Configuration

Infotype configuration is a dictionary with all infotypes at root level key. Each infotype has following configurable parameters (value of each parameter is a dictionary)

- `Prediction_Factors_and_Weights` - This is a dictionary that specifies the weight of each prediction factor which will be used in the final confidence calculation. Following are the prediction factors
  1. Name
  2. Description
  3. Datatype
  4. Values
- `Name` - regex list which is to be matched against column name
- `Description` - regex list which is to be matched against column description
- `Datatype` - list of datatypes to be matched against column datatype
- `Values` - this dictionary contains following information
  1. `prediction_type` - values evaluation model (regex/library)
  2. `regex` - regex list which is to be matched against column values
  3. `library` - library name which is to be used to evaluate column values

#### Sample Infotype Configuration Dictionary

```python
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
```

### Debug Information

A debug information is associated with each infotype proposal, it provides details about confidence score from each prediction factor involved in overall confidence score calculation. This is a dictionary with following four prediction factors as key

- Name
- Description
- Datatype
- Values

```python
{
    'Name': 0.4,
    'Description': 0.2,
    'Values': 0.6,
    'Datatype': 0.3
}
```

### Supported Infotypes

1. Age
2. Gender
3. Person Name / Full Name
4. Email Address
5. Phone Number
6. Street Address
7. Credit-Debit Card Number
8. International Bank Account Number
9. Vehicle Identification Number
10. US Social Security Number
11. Ipv4 Address
12. Ipv6 Address
13. Swift Code
14. US Driving License Number

### Assumptions

- If value prediction factor weight is non-zero (indicating values should be used for infotype inspection) then a minimum 50 non-null column values should be present.

# API 'check_similarity'
This API computes the similarity score between a pair of tables and also between all possible pairs of their constituent columns. Following are the input and output contract:
### API Input
API expects following parameters in the input
- `table_info1` & `table_info2` -  These are instances of TableInfo object. Each TableInfo object contains following class variables:
  - `metadata` - Instance of TableMetadata class containing name, description, platform, table-ID, name_embedding(optional) & desc_embedding(optional) of the table 
  - `parent_tables` - List of table IDs of parent tables of the input table 
  - `column_infos` - List of instances of ColumnInfo object for each constituent column of the table. Each ColumnInfo object has following the class variables 
    - `metadata` - Instance of ColumnMetadata class containing name, description, datatype, dataset_name, column-id, name_embedding(optional) & desc_embedding(optional) of the column 
    - `parent_columns` - List of column IDs of parent columns of the column
- `pruning_mode` - This is a boolean flag, it indicates whether API will run the pruning or non-pruning model. Default value is False (non-pruning mode)
- `use_embeddings` - This is a boolean flag, this tells whether to use meaning based similarity for tables and columns. Default value is False. If this flag is set to True then table/column name and description embeddings needs to be passed in the TableInfo object to API (name/description embeddings can be generated using `preprocess_tables()` API).

### API Output
API returns two objects:
- `table_similarity_score` - This is an instance of SimilarityInfo which contains 'score' and 'prediction_factors_scores'. 'score' is the overall similarity score of the input pair and 'prediction_factors_scores' is the instance of SimilarityDebugInfo class describing the confidence and weighted score of each prediction factor.  
- `column_similarity_scores` - This is a dictionary of similarity scores of all column pairs. Key being a tuple of column IDs of the column pair and value being the instance of SimilarityInfo, it contains the semantic similarity score between two columns and confidence & weighted score of each prediction factor.

### Usage of the API
Find usage of the "check_similarity()" API at following link
https://github.com/mardikark-gslab/datahub-classify/blob/stage_2_dev/datahub-classify/tests/demo_similarity_check.py

### Assumptions
Following are the assumptions about input parameters 
- table_name, table_platform & table_schema are the required parameters in TableInfo object.
- col_name, col_datatype are the required parameters in ColumnInfo object.


# API 'preprocess_tables'
This API generates the embedding for tables (i.e. for name and description of the table and columns)
### API Input
API expects following parameter in the input
- `table_info_list` - This list contains a batch of tables (i.e. instance of TableInfo) for which embedding needs to be populated

### API Output
API returns following parameter
- `table_info_list` - This is a list of TableInfo objects of same length as input. This API populates the embedding of name $ description of the table (and included columns) and updates the same TableInfo and ColumnInfo objects with "name_embedding" and "desc_embedding". Value of "name_embedding" and "desc_embedding" is a list of 'TextEmbeddings' instances.

# Development

### Set up your Python environment

```sh
cd datahub-classify
../gradlew :datahub-classify:installDev # OR pip install -e ".[dev, semantic_similarity]"
source venv/bin/activate
```

### Runnning tests

```sh
pytest tests/ --capture=no --log-cli-level=DEBUG
```

### Sanity check code before committing

```sh
# Assumes: pip install -e ".[dev]" and venv is activated
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/ tests/
```

### Build and Test

```sh
../gradlew :datahub-classify:build
```

You can also run these steps via the gradle build:

```sh
../gradlew :datahub-classify:lint
../gradlew :datahub-classify:lintFix
../gradlew :datahub-classify:testQuick
```
