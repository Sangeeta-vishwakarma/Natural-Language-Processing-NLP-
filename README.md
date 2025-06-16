# Natural-Language-Processing-NLP-

## Natural Language Processing for Sentiment Analysis( Restaurant Reviews )
## Project Overview
This file includes Restaurant reviews dataset and it's code file which executed in jupyter notebook. This notebook demonstrates a classic Natural Language Processing (NLP) pipeline using the Restaurant Reviews Dataset. The goal is to classify customer reviews as positive or negative based on textual sentiment.

### Dataset
- Name: Restaurant_Reviews.tsv
- Format: Tab-separated values (TSV)
- Columns:
  - Review: Text of the customer review  
  - Liked: Binary sentiment label (1 = liked, 0 = disliked)

### Approach
### (i). Text Preprocessing
- Cleaned text using regex to remove non-alphabetic characters.
- Converted to lowercase.
- Tokenized and removed stopwords (but retained "not").
- Applied Stemming using PorterStemmer.
- Built a corpus of cleaned reviews.

### (ii). Feature Extraction
Transformed text into Bag of Words (BoW) using CountVectorizer (max 1500 features).

### (iii). Model Training
- Split data into training and testing sets.
- Used Naive Bayes Classifier (GaussianNB) for binary classification.
- Evaluated using confusion matrix and accuracy score.

### Results
- Accuracy: ~73% (approx, inferred from typical use of this approach).
- Confusion Matrix and classification metrics were used to assess performance.

### Why This Approach?
- Bag of Words is simple yet effective for small/medium datasets.
- Naive Bayes is computationally efficient and performs well in text classification.
- Stemming and selective stopword removal preserve sentiment polarity (e.g., retaining “not” improves context understanding).

### How Could It Be Better?
### 1. Use Lemmatization Instead of Stemming
- Stemming may distort words (e.g., "loved" → "love", but also "worst" → "wor").
- Lemmatization (e.g., with WordNetLemmatizer) is more accurate linguistically.

### 2. Switch to TF-IDF
- CountVectorizer doesn’t consider term importance.
- TfidfVectorizer weighs rare but meaningful words better.

### 3. Model Improvements
- Try Logistic Regression, SVM, or Ensemble Methods like Random Forests or XGBoost for improved accuracy.
- Consider cross-validation for more robust model evaluation.

### 4. Pipeline and Modular Code
- Use sklearn.pipeline.Pipeline for streamlined preprocessing and model training.
- Organize preprocessing as functions or classes for maintainability.


Below Jupyter notebook demonstrates basic Natural Language Processing (NLP) techniques using the NLTK (Natural Language Toolkit) library. It focuses on NLP Preprocessing Techniques which are:
## 1. Customize Remove - Tokenize

- Word tokenization
- Removal of standard English stop words
- Custom stop word filtering based on user-defined lists

### Use Cases:
- Preprocessing before feeding text into machine learning models
- Clean-up of chatbot messages
- Sentiment or review analysis
- Removing domain-specific filler words



## 2. Fetching POS(Parts of Speech)

This script defines a Python function that automates the process of extracting Part-of-Speech (POS) tags for individual words and maps them to WordNet-compatible tags. This is useful in NLP tasks like **lemmatization**, where the correct POS tag significantly improves word normalization.

- Automatically fetch the POS tag of a word using nltk.pos_tag.
- Map the POS tag to WordNet format for downstream tasks like lemmatization.
- Simplify integration with WordNet lemmatizer by converting POS into: wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV

### Use Cases:
- Lemmatization with correct POS tag
- Improving accuracy of text preprocessing
- NLP pipelines (sentiment analysis, question answering, etc.



## 3. Lemmaization - POS

This script demonstrates how to perform **lemmatization** using NLTK’s `WordNetLemmatizer`. Lemmatization is a key step in NLP where words are reduced to their **base or dictionary form**, taking into account the **context** such as the part of speech (POS). This makes it more powerful and accurate than stemming.

- Perform lemmatization on individual words
- Apply lemmatization on a list of words (batch processing)  
- Lemmatize full sentences after tokenization  
- Highlight the effect of different POS tags on lemmatization output

### Why Lemmatization with POS?

Different POS tags give different lemmas for the same word. Improves performance of models in tasks like:
- Sentiment analysis
- Question answering
- Named entity recognition
 -Machine translation



## 4. NER ( Name Entity Recognition )

This script demonstrates the use of **Natural Language Processing (NLP)** techniques such as **Tokenization**, **POS Tagging**, **Named Entity Recognition (NER)**, and **Chunking** using the **NLTK** library. It includes both static text processing and dynamic NER chunking from a real-world corpus (State of the Union addresses).

- Word Tokenization
- POS (Part-of-Speech) Tagging
- Named Entity Recognition (NER)
- Chunking & Tree Visualization
- Sentence Tokenization using an Unsupervised POS-based model

### Why This Matters?
NER helps extract key information like names, organizations, and locations. Useful in applications like:
- Resume parsing
- News summarization
- Information extraction
- Knowledge graph construction



## 5. POS Tag-Tokenize

This script demonstrates **POS-aware lemmatization** using the **NLTK** library. It uses part-of-speech tagging to improve the accuracy of lemmatization by mapping NLTK POS tags to WordNet POS categories before applying lemmatization.

- Word Tokenization  
- POS (Part-of-Speech) Tagging  
- Mapping POS to WordNet Format  
- Context-aware Lemmatization
  
### Why POS-Aware Lemmatization?
Helps lemmatizer understand word context (noun, verb, adjective, adverb). Improves lemmatization accuracy for tasks like:
- Search optimization
- Text classification
- Lemma-based feature engineering in ML pipelines


## 6. POS Tag-Tokenize + Lemmatize

This script demonstrates how to perform **tokenization**, **Part-of-Speech (POS) tagging**, and **lemmatization** on raw text using Python’s NLTK library. It shows how tokenized words can be tagged with their POS and then lemmatized for downstream NLP tasks.
- Word Tokenization  
- POS Tagging using NLTK  
- WordNet Lemmatization (default noun context)
  
### When to Use This?
- Cleaning and preparing raw text data
- POS tagging prior to contextual lemmatization
- Quick analysis or preprocessing for NLP tasks like:
    - Sentiment analysis
    - Chatbot input processing
    - Search engine keyword extraction



## 7. Remove and Customize Stopword

This script demonstrates how to identify and remove **stop words** from a sentence using two popular Python libraries: **NLTK** and **Scikit-learn**. It also covers how to **extend stop word lists** with custom tokens for advanced text preprocessing tasks.
- Stop Word Detection (using NLTK & Scikit-learn)
- Stop Word Removal from text
- Custom Stop Word Extension
  
### Use Cases:
- Reducing noise in text data
- Preprocessing for NLP tasks like topic modeling, sentiment analysis, or text classification
- Adapting stop word lists for domain-specific content (e.g., legal, medical, etc.)



## 8. Stemmer-Types

This script demonstrates how to use **stemming** in Natural Language Processing (NLP) using two popular stemmers provided by the NLTK library:
- **Porter Stemmer**
- **Lancaster Stemmer**
  
Word-level stemming
- List-based stemming
- Sentence tokenization + stemming
- Comparison of different stemmers (Porter vs Lancaster)
  
### Use Cases:
- Keyword normalization for search engines
- Text preprocessing before vectorization
- Matching similar words in noisy datasets



## 9. Using Cosine Similarity
This script calculates the **cosine similarity** between two textual paragraphs using a **manual Bag-of-Words (BoW)** approach. It focuses on comparing content based on domain-specific vocabulary (film domain), and evaluates similarity by vectorizing and comparing word usage.

- Tokenization using `nltk.word_tokenize`
- Stop word removal using NLTK's stopwords
- Set-based Bag-of-Words construction
- Vector representation of text
- Cosine Similarity computation between binary vectors
  
### Use Cases:
- Content similarity checks
- Textual clustering or classification
- Information retrieval and plagiarism detection
- Measuring topic overlap in domain-specific corpora

## 10. Word - Sentence Tokenizer 
This script demonstrates the use of **NLTK (Natural Language Toolkit)** for performing basic Natural Language Processing tasks such as:
- Tokenizing text into words and sentences
- Calculating the frequency distribution of tokens
- Extracting the most common words in a given text
  
### Use Cases:
- Preprocessing text before advanced NLP tasks (e.g., lemmatization, tagging)
- Exploratory text analysis and frequency pattern mining
- Identifying dominant topics or terms in corpora



## 11. Wordnet- Syn,Ant 
This script demonstrates how to use the **WordNet lexical database** via the `nltk.corpus.wordnet` module to extract:
- Synonym sets (synsets)
- Definitions
- Example usages
- Synonyms and antonyms for a given word

NLP Techniques Covered:
- Accessing WordNet synsets
- Extracting word definitions
- Getting example usages of words
- Generating synonym and antonym sets from lemmas

It helps build semantic understanding for NLP tasks like:
- Text generation
- Question answering
- Semantic search
- Sentiment analysis

