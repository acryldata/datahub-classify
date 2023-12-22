from exclude_name_test_config import (
    exclude_name_test_config,
    none_exclude_name_test_config,
)

from datahub_classify.helper_classes import ColumnInfo, Metadata
from datahub_classify.infotype_utils import perform_basic_checks, strip_formatting


def column_infos():
    return [
        ColumnInfo(
            metadata=Metadata(
                meta_info={
                    "Name": "id",
                    "Description": "Unique identifier",
                    "Datatype": "int",
                    "Dataset_Name": "email_data",
                }
            ),
            values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ),
        ColumnInfo(
            metadata=Metadata(
                meta_info={
                    "Name": "email_from",
                    "Description": "Sender's email address",
                    "Datatype": "str",
                    "Dataset_Name": "email_data",
                }
            ),
            values=[
                "bob.myers@gmail.com",
                "jeremy.strong@nyu.edu",
                "alice.smith@yahoo.com",
                "maria.lopez@outlook.com",
                "lucas.grant@edu.org",
                "nora.jones@live.com",
                "ethan.knight@techhub.com",
                "chloe.wilson@health.org",
                "leonard.hayes@finance.com",
                "sophie.turner@arts.com",
            ],
        ),
        ColumnInfo(
            metadata=Metadata(
                meta_info={
                    "Name": "email_to",
                    "Description": "Recipient's email address",
                    "Datatype": "str",
                    "Dataset_Name": "email_data",
                }
            ),
            values=[
                "susan.bones@msn.com",
                "craig.chatterson@datahub.io",
                "john.doe@example.com",
                "kevin.miller@company.com",
                "emily.harris@webmail.com",
                "philip.watson@service.net",
                "diana.ross@music.com",
                "brad.pitts@cinema.com",
                "grace.adams@legalteam.org",
                "rick.martin@sports.net",
            ],
        ),
        ColumnInfo(
            metadata=Metadata(
                meta_info={
                    "Name": "email_sent",
                    "Description": "Indicates if email was sent",
                    "Datatype": "bool",
                    "Dataset_Name": "email_data",
                }
            ),
            values=[False, True, True, False, True, False, True, True, False, True],
        ),
        ColumnInfo(
            metadata=Metadata(
                meta_info={
                    "Name": "email_received",
                    "Description": "Indicates if email was received",
                    "Datatype": "bool",
                    "Dataset_Name": "email_data",
                }
            ),
            values=[False, True, False, False, True, False, False, True, False, False],
        ),
    ]


def test_perform_basic_checks_with_exclude_name():
    for col_data in column_infos():
        result = perform_basic_checks(
            Metadata(col_data.metadata.meta_info),
            col_data.values,
            exclude_name_test_config["Email_Address"],
            "Email_Address",
            1,
        )
        if col_data.metadata.meta_info["Name"] in ["email_sent", "email_received"]:
            assert not result
        else:
            assert result


def test_perform_basic_checks_with_none_exclude_name():
    for col_data in column_infos():
        result = perform_basic_checks(
            Metadata(col_data.metadata.meta_info),
            col_data.values,
            none_exclude_name_test_config["Email_Address"],
            "Email_Address",
            1,
        )
        assert result


def test_strip_formatting():
    assert strip_formatting("Name") == "name"
    assert strip_formatting("my_column_name") == "mycolumnname"
    assert strip_formatting("Col.Name") == "colname"
