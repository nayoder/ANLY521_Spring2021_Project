# Monken and Yoder
# Finding Entity-Labeled Tweets in STS data
# May 17, 2021

import re
import pandas as pd

data_path = '../../data/'

def read_sts(data_file):
    """
    Opens the sts data file and separates the ids, labels, and texts into lists.

    Parameters
    ----------
    data_file : str
        Should be 'sts_gold_tweet.csv'

    Returns
    -------
    ids : list of integers
        IDs for each tweet
    labels : list of integers
        Labels for each tweet (0, 2, or 4)
    texts : list of strings
        Unprocessed tweets

    """
    ids = []
    labels = []
    texts = []
    with open(data_file, 'r') as dd:
        next(dd)  # skip first line, since it is a header
        for line in dd:
            line = line.strip().strip('"')
            line = re.sub(r'";"', ';', line)
            fields = line.split(';')
            ids.append(int(fields[0]))
            labels.append(int(fields[1]))
            texts.append(fields[2])
    return ids, labels, texts


def read_sts2(data_file):
    """
    Opens the sts entities data file and separates the ids, entities, and labels
    into lists.

    Parameters
    ----------
    data_file : str
        Should be 'sts_gold_entity_in_tweet.csv'

    Returns
    -------
    ids : list of integers
        IDs for each tweet
    ents : list of strings
        Entities for each tweet
    labels : list of integers
        Labels for each tweet specific to the entity (0, 2, or 4)
    
    """
    ids = []
    ents = []
    labels = []
    with open(data_file, 'r') as dd:
        next(dd)  # skip first line, since it is a header
        for line in dd:
            line = line.strip().strip('"')
            line = re.sub(r'";"', ';', line)
            fields = line.split(';')
            ids.append(int(fields[0]))
            ents.append(fields[1])
            labels.append(int(fields[2]))
    return ids, ents, labels


if __name__ == '__main__':
    sts_ids, sts_labels, sts_texts = read_sts(data_path + 'sts_gold_tweet.csv')
    ids, ents, labels = read_sts2(data_path + 'sts_gold_entity_in_tweet.csv')
    df1 = pd.DataFrame(list(zip(sts_ids, sts_labels, sts_texts)),
                   columns =['id', 'tweet_label', 'tweet'])
    df2 = pd.DataFrame(list(zip(ids, ents, labels)),
                   columns =['id', 'entity', 'ent_label'])
    # Only the tweets that have entities labeled
    df3 = df1.merge(df2, how='inner')
    df3.to_csv(data_path + 'sts_tweets_entities.csv', index=False)
