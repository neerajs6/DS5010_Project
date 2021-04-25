import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd

class TextAnalysis:
    def __init__(self):


    def Remove_freq_words(words):
        # list of frequent words
        freq_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                      "you'd", 'your', 'yours', 'yourself', 'yourselves',
                      'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
                      'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                      'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
                      'were', 'be', 'been', 'being', 'have', 'has', 'had',
                      'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                      'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                      'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                      'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                      'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
                      'both', 'each', 'few', 'more', 'most', 'other', 'some',
                      'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
                      'will', 'just', 'don', "don't", 'should', "should've",
                      'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                      "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                      'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                      "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
                      "weren't", 'won', "won't", 'wouldn', "wouldn't"]
        # remove frequent words
        tokens_without_frqword = [word for word in words if not word in freq_words]
        return tokens_without_frqword

    def prePro(text):
        cleanedText = Remove_freq_words(text)
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




    def tokenize(s):
        """
        This function preprocesses a string,removing non- alphanumeric
        characters and splits it into separate words at any white spaces.
        :param s: String to be tokenized
        :return: tokenized words separated by any whitespace
        """
        # Initialize empty list for tokenized words using the cleaned text function
        words = []
        # Preprocess text to remove non-alphanumeric characters and maintain spaces
        cleaned = [i for i in s if i.isalnum() or i.isspace()  ]
        cleanedstring = "".join(cleaned)
        # Iterate through s and split strings at every whitespace
        for word in cleanedstring:
            tokens = cleanedstring.split()
            # Use list comprehension with casefold to ensure compatibility for caseless comparisons
            words = [x.casefold() for x in tokens]
        return words


    def count_words(s):
        """

        :param s:
        :return:
        """
        d = {}
        cleanedv1 = [x.casefold() for x in s]
        cleaned = [i for i in cleanedv1 if i.isalnum() or i.isspace()]
        cleanedstring = "".join(cleaned)
        words = cleanedstring.split()

        for word in words:
            if word in d:
                d[word] += 1
            else:
                d[word] = 1
        return d


    #Calculate TF_IDF scores
    def tfidf_scores_plot(self, sample_corpus=None):
        tfIdfVectorizer = TfidfVectorizer(use_idf=False)
        tfIdf = tfIdfVectorizer.fit_transform(sample_corpus)
        df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
        df = df.sort_values('TF-IDF', ascending=True)
        print(df.head(25))

    sample_corpus = ["you are great at coding",
                     "coding is Very fun", "2021 will be much Ntte@#r,"]
    text = prePro(sample_corpus)
    print("preprocessed text = ", sample_corpus)
    tokenized_words = tokenize(sample_corpus)
    print("The tokenized words are ", tokenized_words)
    final_words = Remove_freq_words(tokenized_words)
    print("final list = ", final_words)
