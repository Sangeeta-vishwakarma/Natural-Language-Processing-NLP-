# Natural-Language-Processing-NLP-
This Python script demonstrates basic Natural Language Processing (NLP) techniques using the NLTK (Natural Language Toolkit) library. It focuses on tokenization, stop words removal, and customized filtering for English text.

### Customize Remove - Tokenize

- Word tokenization
- Removal of standard English stop words
- Custom stop word filtering based on user-defined lists

### Fetching POS(Parts of Speech)

This script defines a Python function that automates the process of extracting Part-of-Speech (POS) tags for individual words and maps them to WordNet-compatible tags. This is useful in NLP tasks like **lemmatization**, where the correct POS tag significantly improves word normalization.

- Automatically fetch the POS tag of a word using nltk.pos_tag.

- Map the POS tag to WordNet format for downstream tasks like lemmatization.

- Simplify integration with WordNet lemmatizer by converting POS into: wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV
