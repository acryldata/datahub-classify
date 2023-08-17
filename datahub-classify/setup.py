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
    "vininfo>=1.7.0",
    "schwifty>=2022.9.0",
    "python-stdnum>=1.17",
    "ipaddress>=1.0.23",
    "spacy>=3.4.1,<=3.5.0",
    "phonenumbers>=8.12.56,<=8.13.0",
}

dev_requirements = {
    *base_requirements,
    "black>=22.1.0",
    "coverage>=5.1",
    "flake8>=3.8.3",
    "flake8-tidy-imports>=4.3.0",
    "isort>=5.7.0",
    # Because of https://github.com/python/mypy/issues/13627 issue, set the mypy version to 0.981, we can revisit this in future
    "mypy>=0.981",
    "pytest-cov>=2.8.1",
    "scikit-learn",
    "pandas>=1.2.0,<=1.5.1",
    "openpyxl"
}


setuptools.setup(
    # Package metadata.
    name=package_metadata["__package_name__"],
    version=package_metadata["__version__"],
    url="https://datahubproject.io/",
    project_urls={
        "Documentation": "https://datahubproject.io/docs/",
        "Source": "https://github.com/acryldata/datahub-classify",
        "Changelog": "https://github.com/acryldata/datahub-classify/releases",
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
