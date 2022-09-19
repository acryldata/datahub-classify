# Input Dictionary Format

input1 = {

    'Email_Address': {
        'Prediction_Factors_and_Weights': {
            'Name': 0.3,
            'Description': 0.1,
            'Datatype': 0.1,
            'Values': 0.5
        },
        'Name': {'regex': ["^[e]{0,1}[^a-z0-9]{0,1}mail[s]{0,1}[^a-z0-9]*((add)|(address)){0,1}[^a-z]*$",
                           "mail",
                           ]},
        'Description': {'regex': ["^[e]{0,1}[^a-z0-9]{0,1}mail[s]{0,1}[^a-z0-9]*((add)|(address)){0,1}[^a-z]*$",
                                  "mail",
                                  ]},
        'Datatype': {'type': ['str']},
        'Values': {
            'prediction_type': 'regex',
            'regex': [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
            'library': []
        }
    },

    'Gender': {
        'Prediction_Factors_and_Weights': {
            'Name': 0.3,
            'Description': 0.1,
            'Datatype': 0.1,
            'Values': 0.5
        },
        'Name': {'regex': ["^gender$", "^sex$", "gender", "sex"]},
        'Description': {'regex': ["^gender$", "^sex$", "gender", "sex"]},
        'Datatype': {'type': ['int', 'str']},
        'Values': {
            'prediction_type': 'regex',
            'regex': ["male", "female", "man", "woman", "m", "f"],
            'library': []
        }
    },

    'Credit_Card_Number': {
        'Prediction_Factors_and_Weights': {
            'Name': 0.3,
            'Description': 0.1,
            'Datatype': 0.1,
            'Values': 0.5
        },
        'Name': {'regex': ["^(credit){0,1}[^a-z0-9]{0,1}card[^a-z0-9]*((num)|(no[.]{0,1})|(number)){1}[^a-z]*$",
                           "card"]},
        'Description': {'regex': ["^(credit){0,1}[^a-z0-9]{0,1}card[^a-z0-9]*((num)|(no[.]{0,1})|(number)){1}[^a-z]*$",
                                  "card"]},
        'Datatype': {'type': ['str', 'int']},
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
                      r"^(4903|4905|4911|4936|6333|6759)[0-9]{12}|(4903|4905|4911|4936|6333|6759)[0-9]{14}|(4903|4905|4911|4936|6333|6759)[0-9]{15}|564182[0-9]{10}|564182[0-9]{12}|564182[0-9]{13}|633110[0-9]{10}|633110[0-9]{12}|633110[0-9]{13}$",
                      r"^(6334|6767)[0-9]{12}|(6334|6767)[0-9]{14}|(6334|6767)[0-9]{15}$"
                      ],
            'library': []
        }
    },

    'Phone_Number': {
        'Prediction_Factors_and_Weights': {
            'Name': 0.3,
            'Description': 0.1,
            'Datatype': 0.1,
            'Values': 0.5
        },
        'Name': {'regex': ["^((phone)|(ph[.]{0,1})){1}[^a-z0-9]{0,1}((num)|(no[.]{0,1})|(number)){0,1}[^a-z]*$",
                           "^((mobile)|(mob[.]{0,1})){1}[^a-z0-9]{0,1}((num)|(no[.]{0,1})|(number)){0,1}[^a-z]*$",
                           "^((telephone)|(tel[.]{0,1})){1}[^a-z0-9]{0,1}((num)|(no[.]{0,1})|(number)){0,1}[^a-z]*$",
                           "^(landline){1}[^a-z0-9]{0,1}((num)|(no[.]{0,1})|(number)){0,1}[^a-z]*$",
                           "^(fax){1}[^a-z0-9]{0,1}((num)|(no[.]{0,1})|(number)){0,1}[^a-z]*$",
                           "phone",
                           "telephone",
                           "landline",
                           "mobile",
                           "tel",
                           "fax"]},
        'Description': {'regex': ["^((phone)|(ph[.]{0,1})){1}[^a-z0-9]{0,1}((num)|(no[.]{0,1})|(number)){0,1}[^a-z]*$",
                                  "^((mobile)|(mob[.]{0,1})){1}[^a-z0-9]{0,1}((num)|(no[.]{0,1})|(number)){0,1}[^a-z]*$",
                                  "^((telephone)|(tel[.]{0,1})){1}[^a-z0-9]{0,1}((num)|(no[.]{0,1})|(number)){0,1}[^a-z]*$",
                                  "^(landline){1}[^a-z0-9]{0,1}((num)|(no[.]{0,1})|(number)){0,1}[^a-z]*$",
                                  "^(fax){1}[^a-z0-9]{0,1}((num)|(no[.]{0,1})|(number)){0,1}[^a-z]*$",
                                  "phone",
                                  "telephone",
                                  "landline",
                                  "mobile",
                                  "tel",
                                  "fax"]},
        'Datatype': {'type': ['int', 'str']},
        'Values': {
            'prediction_type': 'library',
            'regex': [],
            'library': ['phonenumbers']
        }
    },

    'Street_Address': {
        'Prediction_Factors_and_Weights': {
            'Name': 0.3,
            'Description': 0.1,
            'Datatype': 0.1,
            'Values': 0.5
        },

        'Name': {'regex': ["^(street){0,1}[^a-z0-9]{0,1}((add)|(address)){1}[^a-z]*$",
                           "^((full)|(complete)){0,1}[^a-z0-9]{0,1}((add)|(address)){1}[^a-z]*$",
                           "^((mail)|(mailing)){0,1}[^a-z0-9]{0,1}((add)|(address)){1}[^a-z]*$",
                           "address",
                           "street",
                           "add"
                           ]},
        'Description': {'regex': ["^(street){0,1}[^a-z0-9]{0,1}((add)|(address)){1}[^a-z]*$",
                                  "^((full)|(complete)){0,1}[^a-z0-9]{0,1}((add)|(address)){1}[^a-z]*$",
                                  "^((mail)|(mailing)){0,1}[^a-z0-9]{0,1}((add)|(address)){1}[^a-z]*$",
                                  "address",
                                  "street",
                                  "add"]},
        'Datatype': {'type': ['str']},
        'Values': {
            'prediction_type': 'library',
            'regex': [],
            'library': ['spacy']
        }
    },

    'Full_Name': {
        'Prediction_Factors_and_Weights': {
            'Name': 0.1,
            'Description': 0.1,
            'Datatype': 0.1,
            'Values': 0.7
        },
        'Name': {'regex': ["^((person)|(user)){0,1}[^a-z0-9]{0,1}((name)){1}[^a-z]*$",
                           "^((person)|(user)){1}[^a-z0-9]{0,1}((name)){0,1}[^a-z]*$",
                           "^(full){0,1}[^a-z0-9]{0,1}((name)){1}[^a-z]*$",
                           "fullname",
                           "name",
                           "person",
                           "user"
                           ]},
        'Description': {'regex': ["^((person)|(user)){0,1}[^a-z0-9]{0,1}((name)){1}[^a-z]*$",
                                  "^((person)|(user)){1}[^a-z0-9]{0,1}((name)){0,1}[^a-z]*$",
                                  "^(full){0,1}[^a-z0-9]{0,1}((name)){1}[^a-z]*$",
                                  "fullname",
                                  "name",
                                  "person",
                                  "user"]},
        'Datatype': {'type': ['str']},
        'Values': {
            'prediction_type': 'library',
            'regex': [],
            'library': []
        }
    },

}
