from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = [line.strip() for line in f]

setup(
    name='adatest',
    version='0.0.2',
    url='https://github.com/microsoft/adatest.git',
    author='Scott Lundberg and Marco Tulio Ribeiro',
    author_email='scott.lundberg@microsoft.com',
    description='Adaptively test and debug any natural language machine learning model.',
    packages=find_packages(exclude=['user_studies', 'notebooks', 'client']),
    install_requires=install_requires
)
