# Monken and Yoder
# Preprocessing STS and JS Tweets
# May 17, 2021

import argparse
import re
import pandas as pd
from nltk.tokenize import TweetTokenizer
import nltk
import demoji

data_path = '../../data/'

def username_list(filename):
    """
    Converts a text file of newline-separated, lowercase usernames into a list
    of usernames.

    Parameters
    ----------
    filename : str
        Either 'sts_usernames.txt' or 'js_usernames.txt'

    Returns
    -------
    un_list : list of strings
    
    """
    with open(filename, 'r') as file:
        un_list = file.read().splitlines() 
    return un_list


def fix_usernames(tweet, ents):
    """
    Removes random usernames (e.g. @user1234) and sets the part of speech for
    the selected entities' usernames (e.g. @POTUS) to NNP.

    Parameters
    ----------
    tweet : list of tuples
        Tweet that has already been tokenized and POS tagged.
    ents : list of strings
        List of selected entities' usernames that should not be removed (e.g. @POTUS)

    Returns
    -------
    tweet_nouser : list of tuples
        Tokenized and POS-tagged tweet with random usernames removed and selected
        entities' usernames tagged as NNP.
        
    """
    tweet_nouser = []
    for token, pos in tweet:
        if token.lower() in ents:
            tweet_nouser.append((token, 'NNP'))
            pass
        if token.startswith('@'):
            pass
        else:
            tweet_nouser.append((token, pos))
    return tweet_nouser



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


def read_js(data_file):
    """
    Opens the js data file and separates the ids and texts into lists.

    Parameters
    ----------
    data_file : str
        Should be 'js_tweets.csv'

    Returns
    -------
    ids : list of integers
        IDs for each tweet
    texts : list of strings
        Unprocessed tweets

    """
    df = pd.read_csv(data_file)
    ids = df.index.tolist()
    texts = df['text'].tolist()
    return ids, texts



def preprocess(texts, ents):
    """
    Preprocesses a tweet by:
        1. Replace emojis with text (e.g. '🦉' becomes 'owl')
        2. Removes urls
        3. Tokenizes using nltk TweetTokenizer
        4. Tags POS using nltk pos_tag
        5. Fixes usernames using fix_usernames function

    Parameters
    ----------
    texts : list of strings
        List of unprocessed tweets
    ents : list of strings
        Usernames of selected entities (e.g. @POTUS)

    Returns
    -------
    tokens_pos : list of lists of tuples
        List of tweets that have been separated into a list of tuples like
        (token, part of speech)

    """
    tk = TweetTokenizer()
    tokens_pos = []
    for text in texts:
        rep_emoji = demoji.replace_with_desc(text, sep=" ")
        rep_http = re.sub(r'https?:\/\/.*\b', '', rep_emoji)
        tokens = tk.tokenize(rep_http)
        tagged = nltk.pos_tag(tokens)
        tagged = fix_usernames(tagged, ents)
        tokens_pos.append(tagged)
    return tokens_pos


def pp_sts(sts_ents):
    """
    Reads sts data, preprocesses the tweets, and writes output to CSV.

    Parameters
    ----------
    sts_ents : list of strings
        Usernames of selected entities (e.g. @POTUS)

    Returns
    -------
    None.

    """
    # Reading data
    sts_ids, sts_labels, sts_texts = read_sts(data_path + 'sts_gold_tweet.csv')
    # Preprocessing tweets
    sts_tokens_pos = preprocess(sts_texts, sts_ents)
    # Writing to CSV
    sts_df = pd.DataFrame(list(zip(sts_ids, sts_tokens_pos, sts_labels)),
               columns =['id', 'tokens_pos', 'label'])
    sts_df.to_csv(data_path + 'sts_tokenized.csv', index=False)


def pp_js(js_ents):
    """
    Reads js data, preprocesses the tweets, and writes output to CSV.

    Parameters
    ----------
    js_ents : list of strings
        Usernames of selected entities (e.g. @POTUS)

    Returns
    -------
    None.

    """
    # Reading data
    js_ids, js_texts = read_js(data_path + 'js_tweets.csv') 
    # Preprocessing tweets for each dataset
    js_tokens_pos = preprocess(js_texts, js_ents)
    # Writing to CSV
    js_df = pd.DataFrame(list(zip(js_ids, js_tokens_pos)),
               columns =['id', 'tokens_pos'])
    js_df.to_csv(data_path + 'js_tokenized.csv', index=False)
    


if __name__ == '__main__':
    # Required for converting emojis to text
    demoji.download_codes()
    
    # Setting up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='js',
                        help="Dataset that should be preprocessed (sts or js)")
    parser.add_argument('--usernames_file', type=str,
                        default='js_usernames.txt',
                        help="Text file of the @usernames of entities (sts_usernames.txt or js_usernames.txt")
    args = parser.parse_args()
    
    # Getting entities from sts_usernames.txt or js_usernames.txt
    ents = username_list(args.usernames_file)
    
    if args.data == 'sts':
        pp_sts(ents)
    elif args.data == 'js':
        pp_js(ents)
    else:
        print("You did not select one of the possible data sources (sts or js).")
