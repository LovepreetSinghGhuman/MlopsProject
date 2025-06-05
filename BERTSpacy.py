#!/usr/bin/env python
# coding: utf-8

# # 1. Import Necessary Libraries
# Standard Library
import csv
import json
import logging
import os
import re
from collections import Counter
from typing import List, Dict, Any, Generator, Tuple, Union, Set

# Third-Party Libraries
import numpy as np
import pandas as pd
import spacy
import fitz
import pdfplumber
from langdetect import detect, DetectorFactory, LangDetectException
import langid
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.io as pio
from bertopic import BERTopic
from concurrent.futures import ThreadPoolExecutor

# spaCy Stopwords
from spacy.lang.en.stop_words import STOP_WORDS as en_stopwords
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stopwords
from spacy.lang.nl.stop_words import STOP_WORDS as nl_stopwords

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# # 2. Extract text from either a single or multiple research study PDFs. Path is 'studies/papers'
# 

# ### Loading functions



# Load spaCy models once
NLP_MODELS = {
    'en': spacy.load("en_core_web_lg"),
    'nl': spacy.load("nl_core_news_lg"),
    'fr': spacy.load("fr_core_news_lg")
}

# Replacement dictionaries
LIGATURES = str.maketrans({
    'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬀ': 'ff', 'ﬅ': 'ft', 'ﬆ': 'st',
    '≤': '<=', '≥': '>=', '≠': '!=', '±': '+-', '→': '->', '∞': 'oo',
    '∫': 'int', '∑': 'sum', '∏': 'prod', '∇': 'nabla', '∂': 'partial', '√': 'sqrt'
})

OCR_FIXES_PATTERN = re.compile(r'signi\s*ficant|di\s*fferent|e\s*ffective|e\s*ffect|chil\s*dren|e\s*ff\s*ective|con\s*fi\s*dence')

REPLACEMENTS = {
    '“': '"', '”': '"', '‘': "'", '’': "'", '—': '-', '–': '-', '…': '...', '•': '*', '·': '*', '●': '*',
    '–': '-', ' %': '%', '^': '', 'kg.': 'kg', 'C°': '°C', 'et. al.': 'et al.', 'et al': 'et al.'
}

UNWANTED_KEYWORDS = frozenset({
    'doi', 'https', 'http', 'journal', 'university', 'copyrighted', 'taylor & francis', 'elsevier',
    'published by', 'received', 'revised', 'author(s)', 'source:', 'history:', 'keywords', 'volume',
    'downloaded', 'article', 'creative commons use', 'authors', 'all rights reserved'
})

REFERENCE_MARKERS = frozenset({'references', 'bibliography', 'acknowledgements', 'method', 'methods'})

# Utility functions
def clean_text(text: str) -> str:
    """Cleans text by replacing ligatures, fixing OCR errors, and normalizing symbols."""
    text = text.translate(LIGATURES)
    text = OCR_FIXES_PATTERN.sub(lambda match: REPLACEMENTS.get(match.group(0), match.group(0)), text)
    for pattern, replacement in REPLACEMENTS.items():
        text = text.replace(pattern, replacement)
    return text

def detect_language(text: str) -> str:
    """Detects the language of the text using langdetect, falling back to langid if needed."""
    try:
        return detect(text)
    except LangDetectException:
        langid_result, _ = langid.classify(text)
        return langid_result if langid_result else 'en'

def extract_tables(pdf_path: str) -> str:
    """Extracts tables using pdfplumber and returns as a formatted string."""
    extracted_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    extracted_text.append("\n".join([" | ".join(row) for row in table if any(row)]))
    except Exception as e:
        logging.error(f"Table extraction failed: {e}")
    return "\n".join(extracted_text)

def process_pdf(file_path: str, filename: str):
    """Processes a PDF file and extracts cleaned text with tables."""
    try:
        with fitz.open(file_path) as pdf_document:
            table_text = extract_tables(file_path)
            for page_num, page in enumerate(pdf_document, start=1):
                text_dict = page.get_text("dict")
                for block in text_dict.get("blocks", []):
                    if block.get("type", 1) != 0:  # Skip images
                        continue

                    paragraph = []
                    prev_x = None
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        line_text = "".join(span["text"] for span in spans)
                        line_text = clean_text(line_text)

                        if any(marker in line_text.lower() for marker in REFERENCE_MARKERS):
                            return  # Stop at references

                        if any(keyword in line_text.lower() for keyword in UNWANTED_KEYWORDS):
                            continue

                        first_word_x = spans[0]["bbox"][0] if spans else 0
                        if prev_x is None or abs(first_word_x - prev_x) < 10:
                            paragraph.append(line_text)
                        else:
                            full_text = " ".join(paragraph).strip()
                            if len(full_text.split()) >= 10:
                                yield [filename, page_num, full_text, detect_language(full_text)]
                            paragraph = [line_text]
                        prev_x = first_word_x

                    if paragraph:
                        full_text = " ".join(paragraph).strip()
                        if len(full_text.split()) >= 10:
                            yield [filename, page_num, full_text, detect_language(full_text)]

            if table_text:  # If tables were found, store them as well
                yield [filename, "tables", table_text, detect_language(table_text)]

    except Exception as e:
        logging.error(f"Failed to process {file_path}: {e}")


# ## The main function that processes the pdf's



# Main function
pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]

with open(CLEANED_CSV, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "page", "text", "language"])

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_pdf, os.path.join(PDF_DIRECTORY, f), f): f for f in pdf_files}

        for future in tqdm(futures, desc="Processing PDFs"):
            for row in future.result():
                writer.writerow(row)

logging.info(f"Data successfully exported to {CLEANED_CSV}")


# ### Showing the cleaned DataFrame



# Load the data
df_cleaned = pd.read_csv(CLEANED_CSV)
df_cleaned.head(30)


# # 3. Clean it using spaCy with language support for english, dutch and french.
# We clean up the text
# - Remove the name of city, country, geography for better outcome
# - Remove special characters (only letters)
# - Convert to lower case
# - Remove stop words
# - Remove words of only one or 2 letters ('a', 'I', at,...)
# - Remove very short sentences
# - Remove urls 
# - use stemming
# - remove duplicate sentences



# Map languages to their respective spaCy models
STOPWORDS_MAP: Dict[str, Set[str]] = {
    'en': en_stopwords,  # Replace with actual stopwords for English
    'fr': fr_stopwords,  # Replace with actual stopwords for French
    'nl': nl_stopwords,  # Replace with actual stopwords for Dutch
}

# Define entity types to remove (Personal Information)
PERSONAL_ENTITIES: Set[str] = {
    "PERSON", "EMAIL", "PHONE", "GPE", "ORG", "NORP", "FAC", "LOC", "PRODUCT", 
    "EVENT", "WORK_OF_ART", "LAW", "DATE"
}

# Regex pattern to remove unwanted characters (e.g., emojis, symbols, numbers, etc.)
UNWANTED_CHARACTERS_PATTERN = re.compile(
    r"[^\w\s"  # Keep alphanumeric characters and whitespace
    r"ÀÁÂÃÄÅàáâãäå"  # Allow common accented characters
    r"ÈÉÊËèéêë" 
    r"ÌÍÎÏìíîï"
    r"ÒÓÔÕÖòóôõö"
    r"ÙÚÛÜùúûü"
    r"ÇçÑñ"  # Allow specific special characters
    r"]", 
    flags=re.UNICODE
)

# Regex pattern to remove numbers
NUMBERS_PATTERN = re.compile(r"\d+")

def preprocess_text(text: str) -> str:
    """
    Preprocesses text by removing unwanted characters, numbers, normalizing spaces, and stripping leading/trailing whitespace.
    Args:
        text (str): Input text to preprocess.
    Returns:
        str: Preprocessed text.
    """
    # Remove unwanted characters using regex
    text = UNWANTED_CHARACTERS_PATTERN.sub("", text)
    # Remove numbers using regex
    text = NUMBERS_PATTERN.sub("", text)
    # Normalize spaces (replace multiple spaces with a single space)
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    return text.strip()

def clean_text_with_spacy(text: str, lang: str) -> str:
    """
    Cleans text using spaCy: removes personal entities, stopwords, and lemmatizes words.
    Args:
        text (str): Input text to clean.
        lang (str): Language code (e.g., 'en', 'fr', 'nl').
    Returns:
        str: Cleaned text.
    """
    if lang not in NLP_MODELS:
        logging.warning(f"Language model for '{lang}' not found. Defaulting to English.")
        lang = 'en'

    nlp = NLP_MODELS.get(lang)
    stopwords = STOPWORDS_MAP.get(lang, set())  # Get stopwords for the language, default to empty set

    if not nlp:
        return text  # If no model is available, return original text

    # Preprocess text to remove unwanted characters and numbers
    text = preprocess_text(text)
    doc = nlp(text)

    # Token processing: lemmatization, stopword removal, personal entity removal
    tokens = [
        token.lemma_.lower() for token in doc
        if token.lemma_  # Ensure lemma exists
        and token.ent_type_ not in PERSONAL_ENTITIES  # Remove personal entities
        and token.text.lower() not in stopwords  # Remove stopwords
        and not token.is_punct  # Remove punctuation
        and not token.is_space  # Remove spaces
        and len(token.lemma_) > 3  # Remove very short words
    ]

    return " ".join(tokens)

def final_clean_csv(input_csv: str):
    """
    Cleans text in a CSV file using spaCy and saves the results.
    Args:
        input_csv (str): Path to the input CSV file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_csv)

        # Validate required columns
        if COLUMN_TEXT not in df or COLUMN_LANGUAGE not in df:
            raise KeyError(f"CSV must contain '{COLUMN_TEXT}' and '{COLUMN_LANGUAGE}' columns")

        # Apply text cleaning in batches to reduce memory usage
        batch_size = 1000  # Adjust based on memory constraints
        cleaned_texts = []

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            cleaned_batch = batch.apply(
                lambda row: clean_text_with_spacy(str(row[COLUMN_TEXT]), row[COLUMN_LANGUAGE]),
                axis=1
            )
            cleaned_texts.extend(cleaned_batch)

        # Add cleaned text to the DataFrame
        df[COLUMN_CLEAN_TEXT] = cleaned_texts

        # Save the cleaned CSV
        df.to_csv(input_csv, index=False)
        logging.info(f"Cleaned data added to new column in {input_csv}")

    except FileNotFoundError:
        logging.error(f"File not found: {input_csv}")
    except KeyError as e:
        logging.error(f"Missing required column in CSV: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing file: {e}")


# Run the cleaning process
final_clean_csv(CLEANED_CSV)



# Display the first 30 rows of the cleaned DataFrame
df_final = pd.read_csv(CLEANED_CSV)
df_final.head(30)



# Check for non-string values to avoid errors before applying topic modeling
print(df_final['clean_text'].isnull().sum())  # Count NaN values
print(df_final[df_final['clean_text'].apply(lambda x: not isinstance(x, str))])  # Find non-string values



# Replace NaN values with empty strings to avoid errors
df_final['clean_text'] = df_final['clean_text'].fillna('')




# Check for non-string values in the 'clean_text' column
print(df_final['clean_text'].isnull().sum())  # Count NaN values
print(df_final[df_final['clean_text'].apply(lambda x: not isinstance(x, str))])  # Find non-string values


# 
# # 4. Initialize and fit BERTopic
# The good thing with BERTopic is that is does most of the work automatically (Meaning, I do not need to bore you to death with details about how it works behind te scenes.)
# 
# We need to do 3 things
# 1. Initialize BERTopic model
# 2. 'Fit' the model -> this  means: run the model, as you would run a simple linear regression
# 3. Look at the topics via 
# 
# To get started, let's just use the default settings.



unique_filenames_count = df_final['filename'].nunique()
print(unique_filenames_count)




# Initialize BERTopic model
topic_model = BERTopic(calculate_probabilities=True, min_topic_size=5, nr_topics=10)

# Fit the model with preprocessed text sentences
topics, probabilities = topic_model.fit_transform(df_final['clean_text'])

# View and inspect topics
topic_model.get_topic_info()