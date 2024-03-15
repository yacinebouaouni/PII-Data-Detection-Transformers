# PII-Data-Detection-Transformers

## Overview
This repository contains scripts for training, validating, and testing transformer-based models for named entity recognition (NER) tasks, specifically for detecting personally identifiable information (PII). The detection and removal of PII from educational data can significantly lower the cost of releasing educational datasets, supporting learning science research and the development of educational tools.

The code is based on the Hugging Face Transformers library and implements a modular approach to detecting PII data and computing the micro-F1 (beta) score. Additionally, it provides functionality to convert tokens between Spacy original tokens and tokens specific to transformers, enabling evaluation in both token formats.

## Features
- Train, validate, cross-validation and test transformer-based models for PII detection
- Modular approach to solution implementation
- Compute micro-F1 (beta) score for evaluation
- Convert tokens between Spacy original tokens and transformer-specific tokens
- Scripts to test with an ensemble of transformers with multiple techniques (averaging, voting ...) 

# Installation

## Prerequisites
- Conda 24.1.2 (miniconda)
- Python 3.11
- Poetry > 1.8 

## Environment Setup
1. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate piidetect

2. Install dependencies using Poetry:
    ```bash
    poetry install
3. Initialize pre-commit hooks for linting and formatting
    ```bash
    poetry run pre-commit install
    ````

## Build the project
```bash
poetry build
```



