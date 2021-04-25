import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
import urllib.request
import heapq
from collections import Counter
import bs4 as BeautifulSoup

class TextSummarization:
    def __init__(self, url):
         self.url=url

    def read_text_url(self,url):
     raw_text= urllib.request.urlopen(url).read() #connecting and reading from url
     text_parse = BeautifulSoup.BeautifulSoup(raw_text,'html.parser') #parsing the html
     text_para = text_parse.find_all('p') #get paragraphs
     text=''
     for t in text_para:  #add all paragraphs to one variable
        text=text+ t.text
     return text


    def prePro(self,text):
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

    #Calculate TF_IDF scores
    # def tfidf_scores_plot(self, sample_corpus=None):
    #     tfIdfVectorizer = TfidfVectorizer(use_idf=False)
    #     tfIdf = tfIdfVectorizer.fit_transform(sample_corpus)
    #     df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
    #     df = df.sort_values('TF-IDF', ascending=True)
    #     print(df.head(25))

    # sample_corpus = ["you are great at coding",
    #                  "coding is Very fun", "2021 will be much Ntte@#r,"]
    # text = prePro(sample_corpus)
    # print("preprocessed text = ", sample_corpus)
    # tokenized_words = tokenize(sample_corpus)
    # print("The tokenized words are ", tokenized_words)
    # final_words = Remove_freq_words(tokenized_words)
    # print("final list = ", final_words)
    
    def tokenize_words(self, text):
      for sentence in text:
        words = text.split()
      
      #Remove common words
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
        
      tokens_without_frqword = [word for word in words if not word in freq_words]
      word_count=Counter(tokens_without_frqword)

      return word_count


    def tokenize_sentences(self, text):
     sentences= re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', text)
     return sentences

    def calc_sentence_scores(self, sentences, word_count):
      sent_score = {}
      for sentence in sentences:
          for word in  sentence.split():
              if word in word_count.keys():
                  if len(sentence.split(' ')) < 28 and len(sentence.split(' ')) > 9: 
                      if sentence not in sent_score.keys():
                          sent_score[sentence] = word_count[word]
                      else:
                          sent_score[sentence] += word_count[word]
      return sent_score

    def generate_summary(self):
      text=self.read_text_url(self.url)
      text=self.prePro(text)
      word_count=self.tokenize_words(text)
      sentences=self.tokenize_sentences(text)
      sent_score=self.calc_sentence_scores(sentences,word_count) 
      summary = heapq.nlargest(round(0.10*len(sentences)), sent_score, key=sent_score.get)
      strx=""
      return (strx.join(summary))

ts=TextSummarization(url='https://en.wikipedia.org/wiki/Data_science#:~:text=Data%20science%20is%20the%20study,analyze%20actual%20phenomena%22%20with%20data.)')
print(ts.generate_summary())
