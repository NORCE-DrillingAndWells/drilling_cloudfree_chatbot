# install required packages 

from setuptools import setup, find_packages

setup(
    name='chatbot_demo',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'requests',
        'numpy',
        'torch',
        'transformers'
    ],
)

