import re
import pandas as pd
from preprocessing import read_sts

def read_sts2(data_file):
    """
    Opens the sts entities data file and separates the ids, entities, and labels
    into lists.

    Parameters
    ----------
    data_file : str
        Should be 'data/sts_gold_entity_in_tweet.csv'

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
    sts_ids, sts_labels, sts_texts = read_sts('data/sts_gold_tweet.csv')
    ids, ents, labels = read_sts2('data/sts_gold_entity_in_tweet.csv')
    df1 = pd.DataFrame(list(zip(sts_ids, sts_labels, sts_texts)),
                   columns =['id', 'tweet_label', 'tweet'])
    df2 = pd.DataFrame(list(zip(ids, ents, labels)),
                   columns =['id', 'entity', 'ent_label'])
    # Only the tweets that have entities labeled
    df3 = df1.merge(df2, how='inner')
    df3.to_csv('data/sts_tweets_entities.csv', index=False)
