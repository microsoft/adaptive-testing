import os
import re
import codecs
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="adaptivetesting",
    version=find_version("adaptivetesting", "__init__.py"),
    url="https://github.com/microsoft/adaptive-testing.git",
    author="Scott Lundberg and Marco Tulio Ribeiro",
    author_email="scott.lundberg@microsoft.com",
    description="Adaptively test and debug any natural language machine learning model.",
    packages=find_packages(exclude=["user_studies", "notebooks", "client"]),
    package_data={"adaptivetesting": ["resources/*"]},
    install_requires=[
        "aiohttp",
        "aiohttp_security",
        "aiohttp_session",
        "appdirs",
        "cryptography",
        "diskcache",
        "nest_asyncio",
        "numpy",
        "pandas",
        "profanity",
        "scikit-learn",
        "shap",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "openai<1",
            "datasets",
            "transformers<4.26",
            "pytest",
            "pytest-mock",
            "torch",
        ]
    },
)
