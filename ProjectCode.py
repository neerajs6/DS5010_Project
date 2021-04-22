import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import stop_words
import numpy as np

def prePro(text):
    cleanedText = text
    cleanedText = re.sub(r'\n', ' ', cleanedText)  # Get rid of new lines replace with spaces
    cleanedText = re.sub(r'(\(([^)^(]+)\))', '',
                         cleanedText)  # removes everything inside of parentheses, have to re-run for nested
    cleanedText = re.sub(r'(\[([^]^[]+)\])', '', cleanedText)  # removes everything inside of square brackets
    cleanedText = re.sub(r'(\{([^}^{]+)\})', '', cleanedText)  # removes everything inside of curly brackets
    cleanedText = re.sub(r'[^\w^\s^.]', ' ',
                         cleanedText)  # Remove all characters not [a-zA-Z0-9_] excluding spaces and periods
    # cleanedText = re.sub(r'\d','', cleanedText) #Remove all numbers
    cleanedText = re.sub(r'(\. ){2,}', '. ', cleanedText).strip()  # Replace all multiple period spaces with one
    return cleanedText