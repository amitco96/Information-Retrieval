import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing
from nltk.stem.porter import *
import math
from nltk.corpus import stopwords
from inverted_index_gcp import *

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", "may", "first", "see", "history", "people", "one", "two", "part", "thumb", "including", "second", "following", "many", "however", "would","became"]
all_stopwords = english_stopwords.union(corpus_stopwords)

# create instance of InvertedIndex

bucket_name = 'index_body83'
inverted_index = InvertedIndex.read_index('.', 'final_index', bucket_name)

class Searching:
    '''
        This class provides functionality for searching through a collection of documents using a specified similarity function
        and a given query.

        Attributes:
            sim_func class: The similarity function(implemented as c class) to be used for calculating document scores.
            query (str): The query string to be searched in the collection of documents.
    '''
    def __init__(self, sim_func, query):
        '''
            Initializes the Searching object with the specified similarity function and query string.
            Args:
                sim_func (function): A similarity function that takes an inverted index as input.
                query (str): The query string to be searched.
        '''
        self.inverted_index = inverted_index
        self.sim_func = sim_func(self.inverted_index)
        self.tokens = self.tokenize_query(query)


    def tokenize_query(self, query):
        '''
            Tokenizes the query string, removes stopwords, and stems the tokens.
            Args:
                query (str): The query string to be tokenized.
            Returns:
                list: A list of stemmed tokens derived from the query.
        '''
        tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
        filtered_tokens = [token for token in tokens if token not in all_stopwords]
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        return stemmed_tokens

    def search(self):
        '''
            Executes the search operation using the provided similarity function and query string.
            Returns:
                list of tuples: A list containing document IDs and their corresponding title based on the search.
        '''
        return self.sim_func.calculate_scores(self.tokens)





class BM25:
    """
        BM25 scoring algorithm for retrieving relevant documents based on a given query.

        Attributes:
            inverted_index (InvertedIndex): An instance of the InvertedIndex class containing the index information.
            k1 (float): Parameter controlling term frequency normalization (default is 1.5).
            b (float): Parameter controlling document length normalization (default is 0.75).
    """

    def __init__(self, inverted_index, k1=1.5, b=0.75):
        """
            Initialize BM25 with the provided parameters.

            Args:
                inverted_index (InvertedIndex): An instance of the InvertedIndex class containing the index information.
                k1 (float, optional): Parameter controlling term frequency normalization (default is 1.5).
                b (float, optional): Parameter controlling document length normalization (default is 0.75).
        """
        self.inverted_index = inverted_index
        self.k1 = k1
        self.b = b

    def calculate_scores(self, query_tokens):
        """
            Calculate BM25 scores for the given query tokens.

            Args:
                query_tokens (list): List of tokens representing the query.

            Returns:
                list of tuples: List of (doc_id, title) tuples for each document.
        """
        # get BM25 parameters
        scores = defaultdict(float)
        N = self.inverted_index.num_of_docs
        avgdl = self.inverted_index.adl

        #calculate idf for each term
        for term in query_tokens:
            if term not in self.inverted_index.df:
                continue
            df = self.inverted_index.df[term]
            idf = math.log2((N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf in self.inverted_index.read_a_posting_list('.', term, bucket_name):
                dl = self.inverted_index.doc_lens[doc_id]
                score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (dl / avgdl)))
                scores[doc_id] += score
        # if there are more than 100 relevant documents return the best 100 ordered by ranking
        if len(scores) > 100:
            new_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
            res = [(str(doc_id), self.inverted_index.id_and_title[doc_id]) for doc_id, _ in new_list]
            return res
        # if there are less than 100 relevant documents return them ordered by ranking
        new_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
        res = [(str(doc_id), self.inverted_index.id_and_title[doc_id]) for doc_id, _ in new_list]
        return res
