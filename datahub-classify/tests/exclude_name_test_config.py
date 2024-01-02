from typing import Any, Dict, List, Union

# Input Dictionary Format

exclude_name_test_config: Dict[
    str, Dict[str, Union[Dict[str, Any], List[str], None]]
] = {
    "Email_Address": {
        "Prediction_Factors_and_Weights": {
            "Name": 1,
            "Description": 0,
            "Datatype": 0,
            "Values": 0,
        },
        "ExcludeName": ["email_sent", "email_received"],
        "Name": {
            "regex": [
                "^.*mail.*id.*$",
                "^.*id.*mail.*$",
                "^.*mail.*add.*$",
                "^.*add.*mail.*$",
                "email",
                "mail",
            ]
        },
        "Description": {"regex": []},
        "Datatype": {"type": ["str"]},
        "Values": {
            "prediction_type": "regex",
            "regex": [],
            "library": [],
        },
    },
}

none_exclude_name_test_config: Dict[str, Dict[str, Union[Dict[str, Any], List[str], None]]] = {  # type: ignore
    "Email_Address": {
        "Prediction_Factors_and_Weights": {
            "Name": 1,
            "Description": 0,
            "Datatype": 0,
            "Values": 0,
        },
        "ExcludeName": None,
        "Name": {
            "regex": [
                "^.*mail.*id.*$",
                "^.*id.*mail.*$",
                "^.*mail.*add.*$",
                "^.*add.*mail.*$",
                "email",
                "mail",
            ]
        },
        "Description": {"regex": []},
        "Datatype": {"type": ["str"]},
        "Values": {
            "prediction_type": "regex",
            "regex": [],
            "library": [],
        },
    },
}
