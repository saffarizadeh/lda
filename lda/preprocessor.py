import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.corpus import wordnet
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class Preprocessor(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stwords = stopwords.words('english') + list(string.punctuation)
        self.table = str.maketrans('', '', string.punctuation)

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    # data cleaning
    def clean_document(self, document):
        document = document.lower()
        document = document.replace(',', ' , ')
        document = document.replace('.', ' . ')
        document = document.replace('-', ' - ')
        document = document.replace('/', ' / ')
        words = document.split()
        filtered_document = ''
        tagged_words = nltk.pos_tag(words)
        for word, tag in tagged_words:
            word = word.translate(self.table)
            tag = self.get_wordnet_pos(tag)
            if word not in self.stwords and len(word) > 1:
                if tag:
                    reduced = self.wnl.lemmatize(word, tag)
                else:
                    reduced = self.wnl.lemmatize(word)
                # reduced = stemmer.stem(word)
                filtered_document = filtered_document + reduced + ' '
        filtered_document = ' '.join(filtered_document.split())
        return filtered_document

    def clean_documents(self, documents):
        cleaned_documents = []
        for document in documents:
            try:
                document = document.strip()
            except:
                pass
            if type(document) == type('') and document != '' and document is not np.nan:
                document = self.clean_document(document)
            else:
                document = 'OOOOOTTTTTT'
            cleaned_documents.append(document)
        return cleaned_documents
