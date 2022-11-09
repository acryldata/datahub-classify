import os
import pathlib

import setuptools

package_metadata: dict = {}
with open("./src/datahub_classify/__init__.py") as fp:
    exec(fp.read(), package_metadata)


def get_long_description():
    root = os.path.dirname(__file__)
    return pathlib.Path(os.path.join(root, "README.md")).read_text(encoding="utf-8")


base_requirements = {
    "spacy==3.4.1",
    "phonenumbers==8.12.56",
    "vininfo==1.7.0",
    "schwifty==2022.9.0",
    "python-stdnum==1.17",
    "ipaddress==1.0.23",
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.0/en_core_web_sm-3.4.0.tar.gz",
}

dev_requirements = {
    *base_requirements,
    "black>=22.1.0",
    "coverage>=5.1",
    "flake8>=3.8.3",
    "flake8-tidy-imports>=4.3.0",
    "isort>=5.7.0",
    "mypy>=0.901,<0.920",
    "pytest-cov>=2.8.1",
    "scikit-learn==1.1.2",
    "pandas==1.5.0",
    "openpyxl"
}


setuptools.setup(
    # Package metadata.
    name=package_metadata["__package_name__"],
    version=package_metadata["__version__"],
    url="https://datahubproject.io/",
    project_urls={
        "Documentation": "https://datahubproject.io/docs/",
        "Source": "https://github.com/datahub-project/datahub",
        "Changelog": "https://github.com/datahub-project/datahub/releases",
    },
    license="Apache License 2.0",
    description="Library to predict info types for DataHub",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
        "Operating System :: POSIX :: Linux",
        "Environment :: Console",
        "Environment :: MacOS X",
        "Topic :: Software Development",
    ],
    # Package info.
    python_requires=">=3.7",
    package_dir={"": "src"},
    packages=setuptools.find_namespace_packages(where="./src"),
    # Dependencies.
    install_requires=list(base_requirements),
    extras_require={"dev": dev_requirements},
)
