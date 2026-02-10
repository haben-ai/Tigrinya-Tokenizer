# Tigriyna_BPE_Tokenizer

A Byte Pair Encoding (BPE) tokenizer for the Tigrinya language, designed for low-resource NLP research and machine learning pipelines.

Tigrinya is a low-resource Semitic language, and most existing tokenizers are optimized for high-resource languages. This project aims to reduce token fragmentation, lower out-of-vocabulary (OOV) rates, and better capture Tigrinya morphology.

---

## Features

- BPE-based subword tokenization for Tigrinya
- Optimized for low-resource settings
- Reduced OOV rate and token fragmentation
- Easy integration into NLP pipelines
- Reproducible tokenizer training and evaluation

---

## Motivation

Tokenization plays a critical role in NLP system performance. Generic tokenizers often perform poorly on Tigrinya due to:

- Rich morphology
- Limited training data
- Underrepresentation in multilingual models

This project addresses these challenges by providing a tokenizer tailored specifically to the Tigrinya language.

---

## Project Structure

```text
Tigriyna_BPE_Tokenizer/
├── data/
│   ├── raw/                 # Raw text data (ignored)
│   ├── processed/           # Processed text data (ignored)
├── tokenizer/
│   ├── train_bpe.py         # Train BPE tokenizer
│   ├── encode.py            # Encode text
│   └── decode.py            # Decode tokens
├── experiments/             # Evaluation and analysis
├── requirements.txt
├── .gitignore
└── README.md
