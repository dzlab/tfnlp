from setuptools import setup
from setuptools import find_packages
import os

# Utility function to read README.md file for long description
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="tfnlp",
    author="dzlab",
    author_email="dzlabs@outlook.com",
    description="An easy-to-use wrapper NLP library for the TensorFlow Models library.",
    url="https://github.com/dzlab/tfnlp",
    keywords="nlp bert tensorflow deep learning",
    license="Apache-2.0",
    version="0.0.1",
    packages=find_packages(),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    install_requires=[
        "tensorflow==2.3.0",
        "tf-models-official==2.3.0"
        ],
    extras_require={
        "test": []
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence :: NLP",
    ],
)