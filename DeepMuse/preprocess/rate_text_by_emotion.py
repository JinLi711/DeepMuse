"""
Rate the emotional level of each sentence in a novel.
"""


import re
from collections import defaultdict
import pandas as pd
import numpy as np
from spacy.lang.en import English
from scipy.signal import argrelextrema
import sys


emolex_path = "../data/NRC_emotion_lexicon_list.txt"
emolex_df = pd.read_csv(
    emolex_path,  
    names=["word", "emotion", "association"], 
    sep='\t'
)
emolex_words = emolex_df.pivot(
    index='word', 
    columns='emotion', 
    values='association'
).reset_index()
emotions = set(emolex_words.columns[1:])

nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))


class EmotionalRating:
    """
    Score the emotional counts of a string.
    
    Emotions are:
        anger
        anticipation
        disgust
        fear
        joy
        sadness
        surprise
        trust
    
    Sentiments are:
        negative
        positive
    """

    emotions = set(emolex_words.columns[1:])

    def __init__(self, string):
        self.raw_string = string
        self.emotion_count = defaultdict(lambda: 0)

    def parse_string(self):
        """
        Lowercase the string.
        Remove punctuation.
        Split the string into a list of words.
        
        :returns: a list of words
        :rtype:   list
        """

        string = self.raw_string
        string = re.sub(r'[^\w\s]', ' ', string.lower())
        string = string.split()
        return string

    def count_emotional_rating(self):
        """
        Increment the counts of emotional word for each word in the 
        sentence.
        """

        def find_value(emotion, word):
            """
            Search the NRC emotion lexicon list.
            
            :param emotion: emotion considered
            :type  emotion: str
            :param word: word to search
            :type  word: str
            """

            try:
                value = emolex_words[emolex_words["word"] ==
                                     word][emotion].values[0]
            except IndexError:
                # word doesn't exist in the emotional chart
                value = 0

            return value

        words = self.parse_string()
        emotional_counts = self.emotion_count

        for word in words:
            for emotion in self.emotions:
                value = find_value(emotion, word)
                emotional_counts[emotion] = emotional_counts[emotion] + value

        self.emotion_count = emotional_counts
        self.total_words = len(words)


class Section:
    """
    Emotional ratings of a section of the literature.
    """

    def __init__(self):
        self.total_words = 0
        self.emotional_counts = 0

    def aggregate_occurences(self, df, start_index, end_index):
        """
        Sum up the emotional and total word counts. 
        """

        section = df[start_index:end_index]
        emotional_counts = {}
        for emotion in emotions:
            emotional_counts[emotion] = section[emotion].sum()

        self.emotional_counts = emotional_counts
        self.total_words = section["word count"].sum()


class Literature:
    """
    Emotional ratings of the entire literature.
    """

    def __init__(self, path):
        self.sections = []
        self.sentence_df = score_literature(path)
        smooth_cum_sum = smooth(
            self.sentence_df["pos - neg cumulative sum"], window_len=25)
        self.smooth_cum_sum = smooth_cum_sum
        self.min_max = find_min_max(smooth_cum_sum)

    def split_into_sections(self):
        min_max_extended = [0] + self.min_max + [sys.maxsize]
        sections = []
        for i, index in enumerate(self.min_max):
            start = min_max_extended[i]
            end = min_max_extended[i + 1]
            section = Section()
            section.aggregate_occurences(self.sentence_df, start, end)
            sections.append(section)
        self.sections = sections


def smooth(x, window_len=11, window='hanning'):
    """
    Smooth the data using a window with requested size.
    
    Source: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """

    if x.ndim != 1:
        raise ValueError(
            "smooth only accepts 1 dimension arrays."
        )

    if x.size < window_len:
        raise ValueError(
            "Input vector needs to be bigger than window size."
        )

    if window_len < 3:
        return x

    window_type = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    if not window in window_type:
        raise ValueError(
            "'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def score_literature(file_path):
    """
    Split the novel into sentences.
    Find emotional ratings of each sentence.
    """

    with open(file_path, 'r') as file:
        text = file.read()

    doc = nlp(text)
    sentences = [sent.string.strip() for sent in doc.sents]

    df_sentences = pd.DataFrame(
        {"sentences": list(filter(lambda x: len(x) != 0, sentences))})
    df_sentences["emotional rating class"] = df_sentences["sentences"].apply(
        lambda x: EmotionalRating(x))
    df_sentences["emotional rating dict"] = df_sentences[
        "emotional rating class"].apply(lambda x: x.count_emotional_rating())
    df_sentences["emotional rating dict"] = df_sentences[
        "emotional rating class"].apply(lambda x: x.emotion_count)
    df_sentences["word count"] = df_sentences["emotional rating class"].apply(
        lambda x: x.total_words)

    for emotion in emotions:
        df_sentences[emotion] = df_sentences["emotional rating dict"].apply(
            lambda x: x[emotion])

    df_sentences[
        "pos - neg"] = df_sentences["positive"] - df_sentences["negative"]

    df_sentences["pos - neg cumulative sum"] = df_sentences[
        "pos - neg"].cumsum()

    return df_sentences


def find_min_max(series):
    """
    Find the local minimum and maximums.
    """
    
    local_max = argrelextrema(series, np.greater)[0]
    local_min = argrelextrema(series, np.less)[0]
    min_max = np.concatenate((local_min, local_max))
    min_max.sort()
    
    return list(min_max)