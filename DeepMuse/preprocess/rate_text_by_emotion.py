"""
Rate the emotional level of each sentence in a novel.
"""


import re
import sys
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import pandas as pd
import numpy as np
from spacy.lang.en import English
from scipy.signal import argrelextrema

from generate_music_specifications import removekey


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


class Literature:
    """
    Emotional ratings of the entire literature.
    """

    def __init__(self, path):
        self.sections = []
        df = score_literature(path)
        self.sentence_df = df
        smooth_cum_sum = smooth(
            self.sentence_df["pos - neg cumulative sum"], window_len=25)
        self.smooth_cum_sum = smooth_cum_sum
        self.min_max = find_min_max(smooth_cum_sum)

        emotion_counts = aggregate_emotional_counts(df)
        emotion_counts["total words"] = df["word count"].sum()
        self.emotion_density = calc_emotion_density(emotion_counts)

    def split_into_sections(self, n_pitches):
        """
        Split the literature into sections based on sharp changes
        of emotional counts.
        """

        min_max_extended = [0] + self.min_max + [sys.maxsize]
        sections = []
        for i, index in enumerate(self.min_max):
            df = self.sentence_df
            start = min_max_extended[i]
            end = min_max_extended[i + 1]
            section = Section(start, end)
            section.aggregate_occurences(df)
            section.create_subsection(df, n_pitches)
            sections.append(section)
        self.sections = sections

    def find_min_max_densities(self, emotion, max_emo=True):
        """
        Given an emotion, find its minimum or maximum
        density over all the sections
        """
        
        densities = []
        sections = self.sections
        for section in sections:
            densities.append(
                section.emotion_density[emotion]
            )

        if max_emo:
            return max(densities)
        else:
            return min(densities)    


class Section:
    """
    Emotional ratings of a section of the literature.
    """

    def __init__(self, start, end):
        self.emotion_counts = dict()
        self.emotion_density = dict()
        self.start = start
        self.end = end
        self.subsections = []

    def aggregate_occurences(self, df):
        """
        Sum up the emotional and total word counts. 
        """

        start_index = self.start
        end_index = self.end
        section = df[start_index:end_index]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = section[emotion].sum()

        emotion_counts["total words"] = section["word count"].sum()
        self.emotion_counts = emotion_counts
        self.emotion_density = calc_emotion_density(emotion_counts)

    def create_subsection(self, df, n_pitches):
        """
        Create emotional densities of subsections of the section.
        """

        subsections = []
        interval = int((self.end - self.start) / n_pitches)

        for i in range(n_pitches):
            start = self.start + i * interval
            end = self.start + (i + 1) * interval
            subsection = Subsection(start, end)
            subsection.aggregate_occurences(df)
            subsections.append(subsection)

        self.subsections = subsections

    def find_min_max_densities(self, emotion, max_emo=True):
        """
        Given an emotion, find its minimum or maximum
        density over all the sections
        """
        
        densities = []
        sections = self.subsections
        for section in sections:
            densities.append(
                section.emotion_density[emotion]
            )

        if max_emo:
            return max(densities)
        else:
            return min(densities) 


class Subsection(Section):
    """
    Subsection of the section.
    Used for determining the pitches.
    """

    pass
    # def __init__(self):
    #     Section.__init__(self)

    # def 

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


def aggregate_emotional_counts(df):
    """
    Given the dataframe, sum up all the counts of the emotions.

    :returns: dictionary mapping emotion to emotion counts
    :rtype:   dict
    """

    emotion_count = dict()

    for emotion in emotions:
        count = df[emotion].sum()
        emotion_count[emotion] = count

    return emotion_count


def find_min_max(series):
    """
    Find the local minimum and maximums.
    """
    
    local_max = argrelextrema(series, np.greater)[0]
    local_min = argrelextrema(series, np.less)[0]
    min_max = np.concatenate((local_min, local_max))
    min_max.sort()
    
    return list(min_max)


def calc_multiple_emotional_counts(mypath):
    """
    Calculate emotional counts for each file.
    
    :returns: dictionary mapping file name to emotion count dict
    :rtype:   dict
    """

    literature_emotions = {}

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in onlyfiles:
        emotion_df = score_literature(mypath + file)
        emotion_counts = aggregate_emotional_counts(emotion_df)
        emotion_counts["total words"] = emotion_df["word count"].sum()
        literature_emotions[file] = emotion_counts

    return literature_emotions


def calc_emotion_density(emotion_dict):
    """
    Given a dict of emotional counts 
    (including the total word counts), 
    calculate the emotion densities.
    """

    emotion_density = dict()

    total_counts = emotion_dict["total words"]
    emotion_dict = removekey(emotion_dict, "total words")
    for emotion, counts in emotion_dict.items():
        emotion_density[emotion] = counts / total_counts

    return emotion_density


def calc_emotion_densities(literature_emotions):
    """
    Calculate the emotional densities for each literature piece.
    
    :returns: dictionary mapping file name to emotion density dict
    :rtype:   dict
    """

    literature_densities = dict()

    for title, emotion_dict in literature_emotions.items():
        literature_densities[title] = calc_emotion_density(emotion_dict)

    return literature_densities


def calc_overall_emotional_density(emo_counts):
    """
    Calculate the overall emotion density.
    Meaning add up all the counts of emotions, divide 
    by the total number of words.
    
    :returns:
    :rtype:   float
    """

    total_words = emo_counts["total words"]

    emo_counts = removekey(emo_counts, "total words")
    total_emo_counts = sum(emo_counts.values())

    return total_emo_counts / total_words