# Monken and Yoder
# SentiCircles
# based on code from here - https://github.com/19shubh/Sentiment-Analysis
# May 17, 2021

import nltk
import matplotlib.pyplot as plt
from nltk.corpus import sentiwordnet as swn
import math
from math import exp, expm1, log, log10
import numpy as np
import turtle
from sklearn import preprocessing
import pandas as pd

def get_TDOC(lines, key, prohib):
    """
    Reads corpus of text, one line per entry and outputs term document correlation

    Parameters
    ----------
    lines : list of strings
        Corpus of text, one line per item
    key : string
        Word of focus
    prohib: list of strings
        words we do not want included in context terms

    Returns
    -------
    norm_radii : dict
        Dictionary of context terms and their TDOC to the key, which is the radius of the senticircle

    """
    freq = {}
    for words in lines:
        if key in words:
            for context in words:
                if context not in prohib and context != key:
                    freq[context] = freq.get(context, 0) + 1

    #### N - total number of terms in corpus
    N = sum([len(line) for line in lines])

    #### N_ci, number of terms where c_i occurs
    Nci = {}
    for context in freq.keys():
        Nci[context] = sum([len(line) for line in lines if context in line])

    #### Radius ... TDOC = f(ci,m) * log(N/Nci)
    radii = {term : freq[term] * (log(N / Nci[term])) for term in freq.keys()}

    #### normalize
    norm = preprocessing.normalize(np.array(list(radii.values())).reshape(1,-1)).reshape(1,-1).tolist()[0]
    norm_radii = {k:norm[i] for i, k in enumerate(radii.keys())}

    return norm_radii  # Returns entire set of context terms related to key


def get_theta(key):
    """
    Creates theta value for senticircles

    Parameters
    ----------
    key : string
        single word for sentiment

    Returns
    -------
    theta : int
        theta value based on sentiment dictionary

    """
    tagged = nltk.pos_tag([key])
    t = tagged[0][0]
    try:
        if 'NN' in tagged[0][1]:
            Scores = swn.senti_synset(t + '.n.01')
        elif 'NNS' in tagged[0][1]:
            Scores = swn.senti_synset(t + '.nns.01')
        elif 'VB' in tagged[0][1]:
            Scores = swn.senti_synset(t + '.v.01')
        elif 'VBG' in tagged[0][1]:
            Scores = swn.senti_synset(t + '.v.01')
        elif 'JJ' in tagged[0][1]:
            Scores = swn.senti_synset(t + '.a.01')
        elif 'RB' in tagged[0][1]:
            Scores = swn.senti_synset(t + '.r.01')
        else:
            return None
    except:
        return None

    if Scores.pos_score() > 0.0:
        return np.pi * Scores.pos_score()
    else:
        return np.pi * Scores.obj_score() * (-1)

def get_xy_coords(radii, theta):
    """ Convert polar coordinates for sentivectors into x-y coordinates"""
    x_y_coords = {}
    for term in theta.keys():
        x_y_coords[term] = (radii[term] * math.cos(theta[term]),radii[term] * math.sin(theta[term]))
    return x_y_coords

def get_sentimedian(x_y_coords):
    """ Calculate median of two-deminsional vectors for sentimedian. """
    return np.mean(np.array(list(x_y_coords.values())), axis=0)

def get_sentiment(sentimedian, lambda_neutral):
    """ Determine categorical sentiment based on lambda neutral value and sentimedian vector. """
    if sentimedian[0] >= 0 and abs(sentimedian[1]) < lambda_neutral:
        return 'neu'
    elif sentimedian[1] > lambda_neutral:
        return 'pos'
    elif sentimedian[1] < - lambda_neutral:
        return 'neg'

def create_plot(xy_coords, key):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot([0, 0], [-1, 1])
    plt.plot([-1, 1], [0, 0])
    plt.axis([-1, 1, -1, 1])
    i = 0
    a = [i[0] for i in xy_coords.values()]
    b = [i[1] for i in xy_coords.values()]

    plt.scatter(a, b, label="circles", color="r", marker="o", s=10)
    q = 0
    names = list(xy_coords.keys())
    for i, j in zip(a, b):
        ax.annotate('%s' % names[q], xy=(i, j), xytext=(15, 0), textcoords='offset points')
        q = q + 1

    ax.add_artist(plt.Circle((0, 0), 1.0, color='b', fill=False))
    plt.xlabel('Sentiment Strength')
    plt.ylabel('Orientation')
    plt.title(key)
    return plt


if __name__ == '__main__':

    # TESTING CODE ONLY

    raw_text = """It's a good movie
    Bad weather cause great problem
    you have a great smile
    android is better than ios
    I am learning sentiment analysis
    this work is tough
    its taking a lot of time.
    mid sems are coming
    i have to study
    this work will have a great effect
    i have to face great problem
    you have a great smile
    all of us has problem
    i like your smile
    99 problem
    """

    data_path = '../../data/'
    dataset = 'sts' # sts
    df = pd.read_pickle(f'{data_path}{dataset}_tokenized.pkl')
    tweets = [x.split(' ') for x in raw_text.split('\n')]
    tweets = df['tokens_pos'].apply(lambda x: [i[0] for i in x])
    key = 'Biden'
    prohib = []
    radii = get_TDOC(tweets,key, prohib)
    theta = {context: get_theta(context) for context in radii.keys()}
    theta = {k : v for k, v in theta.items() if v is not None}
    xy_coords = get_xy_coords(radii, theta)
    sentimedian = get_sentimedian(xy_coords)
    sentiment = get_sentiment(sentimedian, 0.05)
    create_plot(xy_coords, key).show() # next add sentimedian and sentiment to the plot