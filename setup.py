from setuptools import setup, find_packages

setup(
    name='adatest',
    version='0.0.2',
    url='https://github.com/slundberg/adatest.git',
    author='Scott Lundberg and Marco Tulio Ribeiro',
    author_email='scott.lundberg@microsoft.com',
    description='Adaptively test and debug any natural language machine learning model.',
    packages=find_packages(exclude=['user_studies', 'notebooks', 'client']),
    install_requires=['nest_asyncio', 'checklist', 'numpy', 'aiohttp', 'aiohttp_security', 'pandas', 'openai', 'sklearn', 'sentence_transformers'],
)
