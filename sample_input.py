# Input Dictionary Format

input1 = {
    'Email_Address': {
        'Prediction_Factors_and_Weights': {
            'Name': 0.4,
            'Description': 0,
            'Datatype': 0,
            'Values': 0.6
        },
        'Name': {
            'regex': ["^.*mail.*id.*$", "^.*id.*mail.*$",
                      "^.*mail.*add.*$", "^.*add.*mail.*$",
                      "email", "mail"]
        },
        'Description': {
            'regex': ["^.*mail.*id.*$", "^.*mail.*add.*$", "email", "mail"]
        },
        'Datatype': {
            'type': ['str']
        },
        'Values': {
            'prediction_type': 'regex',
            # TODO: Check correctness of /b
            'regex': [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
            'library': []
        }
    },

    'Gender': {
        'Prediction_Factors_and_Weights': {
            'Name': 0.4,
            'Description': 0,
            'Datatype': 0,
            'Values': 0.6
        },
        'Name': {
            'regex': ["^.*gender.*$", "^.*sex.*$", "gender", "sex"]
        },
        'Description': {
            'regex': ["^.*gender.*$", "^.*sex.*$", "gender", "sex"]
        },
        'Datatype': {
            'type': ['int', 'str']
        },
        'Values': {
            'prediction_type': 'regex',
            'regex': ["male", "female", "man", "woman", "m", "f", "w", "men", "women"],
            'library': []
        }
    },
    'Credit_Debit_Card_Number': {
        'Prediction_Factors_and_Weights': {
            'Name': 0.4,
            'Description': 0,
            'Datatype': 0,
            'Values': 0.6
        },
        'Name': {
            'regex': ["^.*card.*number.*$", "^.*number.*card.*$", "^.*credit.*card.*$", "^.*debit.*card.*$"]
        },
        'Description': {
            'regex': ["^.*card.*number.*$", "^.*number.*card.*$", "^.*credit.*card.*$", "^.*debit.*card.*$"]
        },
        'Datatype': {
            'type': ['str', 'int']
        },
        'Values': {
            'prediction_type': 'regex',
            'regex': [r"^4[0-9]{12}(?:[0-9]{3})?$",
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
                      r"^(6334|6767)[0-9]{12}|(6334|6767)[0-9]{14}|(6334|6767)[0-9]{15}$"
                      ],
            'library': []
        }
    },

    'Phone_Number': {
        'Prediction_Factors_and_Weights': {
            'Name': 0.4,
            'Description': 0,
            'Datatype': 0,
            'Values': 0.6
        },
        'Name': {
            'regex': [".*phone.*num.*", ".*num.*phone.*",
                      ".*[^a-z]+ph[^a-z]+.*num.*", ".*num.*[^a-z]+ph[^a-z]+.*",
                      ".*mobile.*num.*", ".*num.*mobile.*",
                      ".*telephone.*num.*", ".*num.*telephone.*",
                      ".*cell.*num.*", ".*num.*cell.*",
                      ".*contact.*num.*", ".*num.*contact.*",
                      ".*landline.*num.*", ".*num.*landline.*",
                      ".*fax.*num.*", ".*num.*fax.*",
                      "phone", "telephone", "landline", "mobile", "tel", "fax", 'cell', 'contact'
                      ]
        },
        'Description': {
            'regex': [".*phone.*num.*", ".*num.*phone.*",
                      ".*[^a-z]+ph[^a-z]+.*num.*", ".*num.*[^a-z]+ph[^a-z]+.*",
                      ".*mobile.*num.*", ".*num.*mobile.*",
                      ".*telephone.*num.*", ".*num.*telephone.*",
                      ".*cell.*num.*", ".*num.*cell.*",
                      ".*contact.*num.*", ".*num.*contact.*",
                      ".*landline.*num.*", ".*num.*landline.*",
                      ".*fax.*num.*", ".*num.*fax.*",
                      "phone", "telephone", "landline", "mobile", "tel", "fax", 'cell', 'contact'
                      ]
        },
        'Datatype': {
            'type': ['int', 'str']
        },
        'Values': {
            'prediction_type': 'library',
            'regex': [],
            'library': ['phonenumbers']
        }
    },

    'Street_Address': {
        'Prediction_Factors_and_Weights': {
            'Name': 0.5,
            'Description': 0,
            'Datatype': 0,
            'Values': 0.5
        },

        'Name': {
            'regex': [".*street.*add.*", ".*add.*street.*",
                      ".*full.*add.*", ".*add.*full.*",
                      ".*mail.*add.*", ".*add.*mail.*",
                      "add[^a-z]+",
                      "address", "street"]
        },
        'Description': {
            'regex': [".*street.*add.*", ".*add.*street.*",
                      ".*full.*add.*", ".*add.*full.*",
                      ".*mail.*add.*", ".*add.*mail.*",
                      "add[^a-z]+",
                      "address", "street"]
        },
        'Datatype': {
            'type': ['str']
        },
        'Values': {
            'prediction_type': 'library',
            'regex': [],
            'library': ['spacy']
        }
    },

    'Full_Name': {
        'Prediction_Factors_and_Weights': {
            'Name': 0.3,
            'Description': 0,
            'Datatype': 0,
            'Values': 0.7
        },
        'Name': {
            'regex': [".*person.*name.*", ".*name.*person.*",
                      ".*user.*name.*", ".*name.*user.*",
                      ".*full.*name.*", ".*name.*full.*",
                      "fullname", "name", "person", "user"]
        },
        'Description': {
            'regex': [".*person.*name.*", ".*name.*person.*",
                      ".*user.*name.*", ".*name.*user.*",
                      ".*full.*name.*", ".*name.*full.*",
                      "fullname", "name", "person", "user"]
        },
        'Datatype': {
            'type': ['str']
        },
        'Values': {
            'prediction_type': 'library',
            'regex': [],
            'library': ['spacy']
        }
    },

    'Age': {
        'Prediction_Factors_and_Weights': {
            'Name': 0.65,
            'Description': 0,
            'Datatype': 0,
            'Values': 0.35
        },
        'Name': {
            'regex': ["age[^a-z]+.*", ".*[^a-z]+age", ".*[^a-z]+age[^a-z]+.*", "age"]
        },
        'Description': {
            'regex': ["age[^a-z]+.*", ".*[^a-z]+age", ".*[^a-z]+age[^a-z]+.*", "age"]
        },
        'Datatype': {
            'type': ['int']
        },
        'Values': {
            'prediction_type': 'library',
            'regex': [],
            'library': ['rule_based_logic']
        }
    }
}
