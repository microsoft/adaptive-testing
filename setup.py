import os
import re
import codecs
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='adatest',
    version=find_version("adatest", "__init__.py"),
    url='https://github.com/microsoft/adatest.git',
    author='Scott Lundberg and Marco Tulio Ribeiro',
    author_email='scott.lundberg@microsoft.com',
    description='Adaptively test and debug any natural language machine learning model.',
    packages=find_packages(exclude=['user_studies', 'notebooks', 'client']),
    package_data={'adatest': ['resources/*']},
    install_requires=install_requires
)
