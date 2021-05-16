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


def combine_unify_tokens(bigram_entities, tagged, synonym_entities):
    """
    Unify token structure across tweets

    Parameters
    ----------
    bigram_entities : list of tuple of bigrams
    tagged : nltk tag list
    synonym_entities : list of tuple of true entity and string options for entity

    Returns
    -------
    clean_tokens : list of strings
        cleaned tokens with combined strings and synonyms
    """
    clean_tokens = []
    skip_next = False
    for i in range(len(tagged)):
        if skip_next == True:  # skip next token if we need to combine bigrams
            skip_next = False
            continue
        if tagged[i][0].lower() in [b[0] for b in
                                    bigram_entities]:  # check if first word of bigram list is matched with token i
            if i + 1 < len(tagged):  # confirm that there is an index for next word
                if tagged[i + 1][0].lower() == \
                        bigram_entities[[b[0] for b in bigram_entities].index(tagged[i][0].lower())][
                            1]:  # check appropriate second word of bigram list that matched previous token
                    clean_tokens.append(tagged[i][0].lower() + '_' + tagged[i + 1][0].lower())  # add combined token
                    skip_next = True  # skip i+1 since it was included here
                else:
                    clean_tokens.append(tagged[i][0].lower())
            else:
                clean_tokens.append(tagged[i][0].lower())
        else:
            clean_tokens.append(tagged[i][0].lower())

    if synonym_entities is not None:
        for entity, options in synonym_entities:
            for i in range(len(clean_tokens)):
                if clean_tokens[i] in options:
                    clean_tokens[i] = entity

    return clean_tokens


def preprocess(texts, ents, bigram_entities, synonym_entities):
    """
    Preprocesses a tweet by:
        1. Replace emojis with text (e.g. '🦉' becomes 'owl')
        2. Removes urls
        3. Tokenizes using nltk TweetTokenizer
        4. Tags POS using nltk pos_tag
        5. Fixes usernames using fix_usernames function
        6. Converts tokens to bigram tokens
        7. Converts synonym tokens to single form

    Parameters
    ----------
    texts : list of strings
        List of unprocessed tweets
    ents : list of strings
        Usernames of selected entities (e.g. @POTUS)
    bigram_entities : list of tuple of strings
        tuple of bigrams to combine
    synonym_entities : list of tuple
        0th item of tuple is standard form, 1st item of tuple is list of strings for synonym options to combine
    Returns
    -------
    tokens_plain : list of lists of strings
        List of tweets that have been separated into a list of strings
    tokens_pos : list of lists of tuples
        List of tweets that have been separated into a list of tuples like
        (token, part of speech)

    """
    tk = TweetTokenizer(reduce_len=True) # reduce long strings

    tokens_plain = []
    tokens_pos = []
    for text in texts:
        rep_emoji = demoji.replace_with_desc(text, sep=" ")
        rep_http = re.sub(r'https?:\/\/.*\b', '', rep_emoji)
        tokens = tk.tokenize(rep_http)
        tagged = nltk.pos_tag(tokens)
        tagged = fix_usernames(tagged, ents)

        clean_tokens = combine_unify_tokens(bigram_entities, tagged, synonym_entities)

        tokens_plain.append(clean_tokens)
        tokens_pos.append(tagged)

    return tokens_plain, tokens_pos





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
    sts_df = pd.DataFrame(list(zip(sts_ids, sts_texts, sts_labels)),
               columns =['id', 'text', 'tweet_label'])

    sts_entity = pd.read_csv(data_path + 'sts_gold_entity_in_tweet.csv', sep = ';')
    sts_entity_agg = pd.read_csv(data_path + 'sts_gold_entity_aggregated.csv', sep = ';')

    # determine sentiment label for entities
    def determine_sentiment(x):

        if x['#neutral'] >= x['#negative'] and x['#neutral'] >= x['#positive']:
            return 'neutral'
        elif x['#positive'] > x['#negative']:
            return 'positive'
        elif x['#negative'] > x['#positive']:
            return 'negative'
        elif x['#negative'] == x['#positive']:
            return 'neutral'

    sts_entity_agg['sentiment'] = sts_entity_agg.apply(lambda x: determine_sentiment(x), axis = 1)
    sts_entity_agg = sts_entity_agg[['entity','sentiment']]

    # create bigram entities for appropriate entity recognition
    bigram_entities = sts_entity_agg.entity[sts_entity_agg.entity.str.contains('_')].str.split('_').tolist()

    # subset tweets to only ones with entity-level labels
    sts_df = sts_df[sts_df['id'].isin(sts_entity['tweet_id'].tolist())].reset_index(drop = True)

    # create tokens from raw text, deal with bigrams, and apply synonyms
    sts_df['tokens_plain'], sts_df['tokens_pos'] = preprocess(sts_df['text'], sts_ents, bigram_entities, synonym_entities = None)

    # Writing to CSV
    sts_df.to_pickle(data_path + 'sts_tokenized.pkl')

    sts_entity_agg.to_pickle(data_path + 'sts_labels.pkl')


def pp_js(js_ents, bigrams_file, synonyms_file):
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

    js_bigrams = [x.replace('\n','').split(',') for x in open(bigrams_file,'r').readlines()]
    js_synonyms = [x.replace('\n','').split(',') for x in open(synonyms_file,'r').readlines()]
    js_synonyms = [[x[0],x[1:]] for x in js_synonyms]

    # Preprocessing tweets for each dataset
    js_tokens_plain, js_tokens_pos = preprocess(js_texts, js_ents, js_bigrams, js_synonyms)
    # Writing to CSV
    js_df = pd.DataFrame(list(zip(js_ids, js_tokens_plain, js_tokens_pos)),
               columns =['id', 'tokens_plain', 'tokens_pos'])
    js_df.to_pickle(data_path + 'js_tokenized.pkl')
    


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
    parser.add_argument('--bigrams_file', type=str,
                        default='js_bigrams.txt',
                        help="text file with the bigrams for individual tokens to combine into bigrams")
    parser.add_argument('--synonyms_file', type=str,
                        default='js_synonyms.txt',
                        help="text file with a standard surface form and alternate forms of tokens in order to unify across texts")
    args = parser.parse_known_args()[0]
    
    # Getting entities from sts_usernames.txt or js_usernames.txt
    ents = username_list(args.usernames_file)
    
    if args.data == 'sts':
        pp_sts(ents)
    elif args.data == 'js':
        pp_js(ents, args.bigrams_file, args.synonyms_file)
    else:
        print("You did not select one of the possible data sources (sts or js).")
