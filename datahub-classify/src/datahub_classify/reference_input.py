from typing import Any, Dict, List, Union

# Input Dictionary Format

input1: Dict[str, Dict[str, Union[Dict[str, Any], List[str], None]]] = {
    "Email_Address": {
        "Prediction_Factors_and_Weights": {
            "Name": 0.4,
            "Description": 0,
            "Datatype": 0,
            "Values": 0.6,
        },
        "ExcludeName": [],
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
        "Description": {
            "regex": ["^.*mail.*id.*$", "^.*mail.*add.*$", "email", "mail"]
        },
        "Datatype": {"type": ["str"]},
        "Values": {
            "prediction_type": "regex",
            "regex": [r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"],
            "library": [],
        },
    },
    "Gender": {
        "Prediction_Factors_and_Weights": {
            "Name": 0.4,
            "Description": 0,
            "Datatype": 0,
            "Values": 0.6,
        },
        "Name": {"regex": ["^.*gender.*$", "^.*sex.*$", "gender", "sex"]},
        "Description": {"regex": ["^.*gender.*$", "^.*sex.*$", "gender", "sex"]},
        "Datatype": {"type": ["int", "str"]},
        "Values": {
            "prediction_type": "regex",
            "regex": ["male", "female", "man", "woman", "m", "f", "w", "men", "women"],
            "library": [],
        },
    },
    "Credit_Debit_Card_Number": {
        "Prediction_Factors_and_Weights": {
            "Name": 0.4,
            "Description": 0,
            "Datatype": 0,
            "Values": 0.6,
        },
        "Name": {
            "regex": [
                "^.*card.*number.*$",
                "^.*number.*card.*$",
                "^.*credit.*card.*$",
                "^.*debit.*card.*$",
                "ccn[^a-z]+.*",
                ".*[^a-z]+ccn",
                ".*[^a-z]+ccn[^a-z]+.*",
                "ccn",
            ]
        },
        "Description": {
            "regex": [
                "^.*card.*number.*$",
                "^.*number.*card.*$",
                "^.*credit.*card.*$",
                "^.*debit.*card.*$",
                "ccn[^a-z]+.*",
                ".*[^a-z]+ccn",
                ".*[^a-z]+ccn[^a-z]+.*",
                "ccn",
            ]
        },
        "Datatype": {"type": ["str", "int"]},
        "Values": {
            "prediction_type": "regex",
            "regex": [
                r"^4[0-9]{12}(?:[0-9]{3})?$",
                r"^(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}$",
                r"^3[47][0-9]{13}$",
                r"^3(?:0[0-5]|[68][0-9])[0-9]{11}$",
                r"^6(?:011|5[0-9]{2})[0-9]{12}$",
                r"^(?:2131|1800|35\d{3})\d{11}$",
                r"^(6541|6556)[0-9]{12}$",
                r"^389[0-9]{11}$",
                r"^63[7-9][0-9]{13}$",
                r"^9[0-9]{15}$",
                r"^(6304|6706|6709|6771)[0-9]{12,15}$",
                r"^(5018|5020|5038|6304|6759|6761|6763)[0-9]{8,15}$",
                r"^(62[0-9]{14,17})$",
                r"^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})$",
                r"^(4903|4905|4911|4936|6333|6759)[0-9]{12}|(4903|4905|4911|4936|"
                r"6333|6759)[0-9]{14}|(4903|4905|4911|4936|6333|6759)[0-9]{15}|564182"
                r"[0-9]{10}|564182[0-9]{12}|564182[0-9]{13}|633110[0-9]{10}|633110[0-9]"
                r"{12}|633110[0-9]{13}$",
                r"^(6334|6767)[0-9]{12}|(6334|6767)[0-9]{14}|(6334|6767)[0-9]{15}$",
            ],
            "library": [],
        },
    },
    "Phone_Number": {
        "Prediction_Factors_and_Weights": {
            "Name": 0.4,
            "Description": 0,
            "Datatype": 0,
            "Values": 0.6,
        },
        "Name": {
            "regex": [
                ".*phone.*(num|no).*",
                ".*(num|no).*phone.*",
                ".*[^a-z]+ph[^a-z]+.*(num|no).*",
                ".*(num|no).*[^a-z]+ph[^a-z]+.*",
                ".*mobile.*(num|no).*",
                ".*(num|no).*mobile.*",
                ".*telephone.*(num|no).*",
                ".*(num|no).*telephone.*",
                ".*cell.*(num|no).*",
                ".*(num|no).*cell.*",
                ".*contact.*(num|no).*",
                ".*(num|no).*contact.*",
                ".*landline.*(num|no).*",
                ".*(num|no).*landline.*",
                ".*fax.*(num|no).*",
                ".*(num|no).*fax.*",
                "phone",
                "telephone",
                "landline",
                "mobile",
                "tel",
                "fax",
                "cell",
                "contact",
            ]
        },
        "Description": {
            "regex": [
                ".*phone.*(num|no).*",
                ".*(num|no).*phone.*",
                ".*[^a-z]+ph[^a-z]+.*(num|no).*",
                ".*(num|no).*[^a-z]+ph[^a-z]+.*",
                ".*mobile.*(num|no).*",
                ".*(num|no).*mobile.*",
                ".*telephone.*(num|no).*",
                ".*(num|no).*telephone.*",
                ".*cell.*(num|no).*",
                ".*(num|no).*cell.*",
                ".*contact.*(num|no).*",
                ".*(num|no).*contact.*",
                ".*landline.*(num|no).*",
                ".*(num|no).*landline.*",
                ".*fax.*(num|no).*",
                ".*(num|no).*fax.*",
                "phone",
                "telephone",
                "landline",
                "mobile",
                "tel",
                "fax",
                "cell",
                "contact",
            ]
        },
        "Datatype": {"type": ["int", "str"]},
        "Values": {
            "prediction_type": "library",
            "regex": [],
            "library": ["phonenumbers"],
        },
    },
    "Street_Address": {
        "Prediction_Factors_and_Weights": {
            "Name": 0.5,
            "Description": 0,
            "Datatype": 0,
            "Values": 0.5,
        },
        "Name": {
            "regex": [
                ".*street.*add.*",
                ".*add.*street.*",
                ".*full.*add.*",
                ".*add.*full.*",
                ".*mail.*add.*",
                ".*add.*mail.*",
                "add[^a-z]+",
                "address",
                "street",
            ]
        },
        "Description": {
            "regex": [
                ".*street.*add.*",
                ".*add.*street.*",
                ".*full.*add.*",
                ".*add.*full.*",
                ".*mail.*add.*",
                ".*add.*mail.*",
                "add[^a-z]+",
                "address",
                "street",
            ]
        },
        "Datatype": {"type": ["str"]},
        "Values": {"prediction_type": "library", "regex": [], "library": ["spacy"]},
    },
    "Full_Name": {
        "Prediction_Factors_and_Weights": {
            "Name": 0.3,
            "Description": 0,
            "Datatype": 0,
            "Values": 0.7,
        },
        "Name": {
            "regex": [
                ".*person.*name.*",
                ".*name.*person.*",
                ".*user.*name.*",
                ".*name.*user.*",
                ".*full.*name.*",
                ".*name.*full.*",
                "fullname",
                "name",
                "person",
                "user",
            ]
        },
        "Description": {
            "regex": [
                ".*person.*name.*",
                ".*name.*person.*",
                ".*user.*name.*",
                ".*name.*user.*",
                ".*full.*name.*",
                ".*name.*full.*",
                "fullname",
                "name",
                "person",
                "user",
            ]
        },
        "Datatype": {"type": ["str"]},
        "Values": {"prediction_type": "library", "regex": [], "library": ["spacy"]},
    },
    "Age": {
        "Prediction_Factors_and_Weights": {
            "Name": 0.65,
            "Description": 0,
            "Datatype": 0,
            "Values": 0.35,
        },
        "Name": {
            "regex": ["age[^a-z]+.*", ".*[^a-z]+age", ".*[^a-z]+age[^a-z]+.*", "age"]
        },
        "Description": {
            "regex": ["age[^a-z]+.*", ".*[^a-z]+age", ".*[^a-z]+age[^a-z]+.*", "age"]
        },
        "Datatype": {"type": ["int"]},
        "Values": {
            "prediction_type": "library",
            "regex": [],
            "library": ["rule_based_logic"],
        },
    },
    "IBAN": {
        "Prediction_Factors_and_Weights": {
            "Name": 0.4,
            "Description": 0,
            "Datatype": 0,
            "Values": 0.6,
        },
        "Name": {
            "regex": [
                "iban[^a-z]+.*",
                ".*[^a-z]+iban",
                ".*[^a-z]+iban[^a-z]+.*",
                ".*int.*bank.*acc.*",
                "iban",
            ]
        },
        "Description": {
            "regex": [
                "iban[^a-z]+.*",
                ".*[^a-z]+iban",
                ".*[^a-z]+iban[^a-z]+.*",
                ".*int.*bank.*acc.*",
                "iban",
            ]
        },
        "Datatype": {"type": ["int"]},
        "Values": {"prediction_type": "library", "regex": [], "library": ["scwifty"]},
    },
    "US_Social_Security_Number": {
        "Prediction_Factors_and_Weights": {
            "Name": 0.4,
            "Description": 0,
            "Datatype": 0,
            "Values": 0.6,
        },
        "Name": {
            "regex": [
                "ssn[^a-z]+.*",
                ".*[^a-z]+ssn",
                ".*[^a-z]+ssn[^a-z]+.*",
                ".*social.*security.*",
                ".*us.*identification.*(num|no).*",
                ".*identification.*(num|no).*us.*",
                ".*united.*states.*identification.*(num|no).*",
                ".*identification.*(num|no).*united.*states.*",
                "social",
                "security",
                "ssn",
            ]
        },
        "Description": {
            "regex": [
                "ssn[^a-z]+.*",
                ".*[^a-z]+ssn",
                ".*[^a-z]+ssn[^a-z]+.*",
                ".*social.*security.*",
                ".*us.*identification.*(num|no).*",
                ".*identification.*(num|no).*us.*",
                ".*united.*states.*identification.*(num|no).*",
                ".*identification.*(num|no).*united.*states.*",
                "social",
                "security",
                "ssn",
            ]
        },
        "Datatype": {"type": ["int", "str"]},
        "Values": {"prediction_type": "library", "regex": [], "library": ["stdnum"]},
    },
    "Vehicle_Identification_Number": {
        "Prediction_Factors_and_Weights": {
            "Name": 0.4,
            "Description": 0,
            "Datatype": 0,
            "Values": 0.6,
        },
        "Name": {
            "regex": [
                "vin[^a-z]+.*",
                ".*[^a-z]+vin",
                ".*[^a-z]+vin[^a-z]+.*",
                ".*vehicle.*identification.*",
                ".*chassis.*(num|no).*",
                ".*frame.*(num|no).*",
                "vin",
            ]
        },
        "Description": {
            "regex": [
                "vin[^a-z]+.*",
                ".*[^a-z]+vin",
                ".*[^a-z]+vin[^a-z]+.*",
                ".*vehicle.*identification.*",
                ".*chassis.*(num|no).*",
                ".*frame.*(num|no).*",
            ]
        },
        "Datatype": {"type": ["str"]},
        "Values": {"prediction_type": "library", "regex": [], "library": ["vininfo"]},
    },
    "IP_Address_v4": {
        "Prediction_Factors_and_Weights": {
            "Name": 0.4,
            "Description": 0,
            "Datatype": 0,
            "Values": 0.6,
        },
        "Name": {
            "regex": [
                "ip[^a-z]+.*",
                ".*[^a-z]+ip",
                ".*[^a-z]+ip[^a-z]+.*",
                "ip[^a-z]{0,1}v4.*",
                ".*[^a-z]+ip[^a-z]{0,1}v4.*",
                ".*ip.*address.*",
                ".*source.*add.*",
                ".*add.*source.*",
                ".*destination.*add.*",
                ".*add.*destination.*",
                ".*src.*add.*",
                ".*add.*src.*",
                ".*d[e]{0,1}st.*add.*",
                ".*add.*d[e]{0,1}st.*",
                "ip",
            ]
        },
        "Description": {
            "regex": [
                "ip[^a-z]+.*",
                ".*[^a-z]+ip",
                ".*[^a-z]+ip[^a-z]+.*",
                "ip[^a-z]{0,1}v4.*",
                ".*[^a-z]+ip[^a-z]{0,1}v4.*",
                ".*ip.*address.*",
                ".*source.*add.*",
                ".*add.*source.*",
                ".*destination.*add.*",
                ".*add.*destination.*",
                ".*src.*add.*",
                ".*add.*src.*",
                ".*d[e]{0,1}st.*add.*",
                ".*add.*d[e]{0,1}st.*",
                "ip",
            ]
        },
        "Datatype": {"type": ["str"]},
        "Values": {
            "prediction_type": "library",
            "regex": [
                r"^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])$"
            ],
            "library": ["ipaddress"],
        },
    },
    "IP_Address_v6": {
        "Prediction_Factors_and_Weights": {
            "Name": 0.4,
            "Description": 0,
            "Datatype": 0,
            "Values": 0.6,
        },
        "Name": {
            "regex": [
                "ip[^a-z]+.*",
                ".*[^a-z]+ip",
                ".*[^a-z]+ip[^a-z]+.*",
                "ip[^a-z]{0,1}v6.*",
                ".*[^a-z]+ip[^a-z]{0,1}v6.*",
                ".*ip.*address.*",
                ".*source.*add.*",
                ".*add.*source.*",
                ".*destination.*add.*",
                ".*add.*destination.*",
                ".*src.*add.*",
                ".*add.*src.*",
                ".*d[e]{0,1}st.*add.*",
                ".*add.*d[e]{0,1}st.*",
                "ip",
            ]
        },
        "Description": {
            "regex": [
                "ip[^a-z]+.*",
                ".*[^a-z]+ip",
                ".*[^a-z]+ip[^a-z]+.*",
                "ip[^a-z]{0,1}v6.*",
                ".*[^a-z]+ip[^a-z]{0,1}v6.*",
                ".*ip.*address.*",
                ".*source.*add.*",
                ".*add.*source.*",
                ".*destination.*add.*",
                ".*add.*destination.*",
                ".*src.*add.*",
                ".*add.*src.*",
                ".*d[e]{0,1}st.*add.*",
                ".*add.*d[e]{0,1}st.*",
                "ip",
            ]
        },
        "Datatype": {"type": ["str"]},
        "Values": {
            "prediction_type": "library",
            "regex": [
                r"(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))"
            ],
            "library": ["ipaddress"],
        },
    },
    "US_Driving_License_Number": {
        "Prediction_Factors_and_Weights": {
            "Name": 0.4,
            "Description": 0,
            "Datatype": 0,
            "Values": 0.6,
        },
        "Name": {
            "regex": [
                "dl[^a-z]+.*",
                ".*[^a-z]+dl",
                ".*[^a-z]+dl[^a-z]+.*",
                ".*license.*(num|no).*",
                ".*(num|no).*license.*",
                ".*driv.*lic.*",
                ".*lic.*driv.*",
                "driv",
                "lic",
            ]
        },
        "Description": {
            "regex": [
                "dl[^a-z]+.*",
                ".*[^a-z]+dl",
                ".*[^a-z]+dl[^a-z]+.*",
                ".*license.*(num|no).*",
                ".*(num|no).*license.*",
                ".*driv.*lic.*",
                ".*lic.*driv.*",
                "driv",
                "lic",
            ]
        },
        "Datatype": {"type": ["str"]},
        "Values": {
            "prediction_type": "regex",
            "regex": [
                r"\d{7}",
                r"[a-zA-Z]\d{8}",
                r"\d{9}",
                r"9\d{8}",
                r"[a-zA-Z]\d{7}",
                r"\d{2}-\d{3}-\d{4}",
                r"[a-zA-Z] \d{3} \d{3} \d{3} \d{3}",
                r"[a-zA-Z]\d{12}",
                r"[a-zA-Z]\d{3}-\d{3}-\d{2}-\d{3}-\d",
                r"[a-zA-Z]-\d{3}-\d{3}-\d{3}-\d{3}",
                r"[a-zA-Z]\s\d{3}\s\d{3}\s\d{3}\s\d{3}",
                r"[a-zA-Z]\d{12}",
                r"[a-zA-Z]{2}\d{6}[a-zA-Z]",
                r"[a-zA-Z]\d{3}-\d{4}-\d{4}",
                r"[a-zA-Z]\d{11}",
                r"\d{4}-\d{2}-\d{4}",
                r"\d{3}[a-zA-Z]{2}\d{4}",
                r"[a-zA-Z]\d{2}-\d{2}-\d{4}",
                r"[a-zA-Z]\d{2}-\d{3}-\d{3}",
                r"[a-zA-Z]\d{9}",
                r"\d{3}-\d{2}-\d{4}",
                r"[a-zA-Z]\d{9}",
                r"(([0][1-9]|[1][0-2])\d{3}([1-9][0-9]{3})41([0][1-9]|[1][0-9]|[3][0-1]))"
                r"\d{10}",
                r"([0][1-9]|[1][0-2])[a-zA-Z]{3}\d{2}(0[1-9]|[1-2][0-9]|3[0-1])\d",
                r"[a-zA-Z]\d{4}-\d{5}-\d{5}",
                r"[a-zA-Z]\d{14}",
                r"\d{3} \d{3} \d{3}",
                r"\d{12}",
                r"[a-zA-Z]{3}-\d{2}-\d{4}",
                r"[a-zA-Z]{1}[0-9]{4,8}",
                r"[a-zA-Z]{2}[0-9]{3,7}",
                r"\d{2}\s\d{3}\s\d{3}",
                r"[1-9]{2}\d{5}",
                r"\d{8}",
                r"\d{7,9}",
                r"\d{7}[a-zA-Z]",
                r"[a-zA-Z]\d{8}",
                r"[a-zA-Z*]{1,6}[a-zA-Z]{2}\d[a-zA-Z0-9]{4,6}",
                r"[a-zA-Z]\d{6}",
                r"[a-zA-Z]\d{3}-\d{4}-\d{4}-\d{2}",
                r"\d{6}-\d{3}",
                r"\d{2}[a-zA-Z]{3}\d{5}",
                r"\d{13}",
            ],
            "library": [],
        },
    },
    "Swift_Code": {
        "Prediction_Factors_and_Weights": {
            "Name": 0.4,
            "Description": 0,
            "Datatype": 0,
            "Values": 0.6,
        },
        "Name": {
            "regex": [
                "bic[^a-z]+.*",
                ".*[^a-z]+bic",
                ".*[^a-z]+bic[^a-z]+.*",
                ".*swift.*(num|no).*",
                ".*(num|no).*swift.*",
                ".*swift.*code.*",
                ".*code.*swift.*",
                ".*swift.*",
                ".*business.*identifier.*code.*",
                "bic",
            ]
        },
        "Description": {
            "regex": [
                "bic[^a-z]+.*",
                ".*[^a-z]+bic",
                ".*[^a-z]+bic[^a-z]+.*",
                ".*swift.*(num|no).*",
                ".*(num|no).*swift.*",
                ".*swift.*code.*",
                ".*code.*swift.*",
                ".*swift.*",
                ".*business.*identifier.*code.*",
                "bic",
            ]
        },
        "Datatype": {"type": ["int", "str"]},
        "Values": {"prediction_type": "library", "regex": [], "library": ["schwifty"]},
    },
}
