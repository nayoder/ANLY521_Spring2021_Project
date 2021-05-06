import re
import pandas as pd
from nltk.tokenize import TweetTokenizer
import nltk
import demoji



def fix_usernames(tweet, ents):
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
    df = pd.read_csv(data_file)
    ids = df.index.tolist()
    texts = df['text'].tolist()
    return ids, texts



def preprocess(texts, ents):
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
    



if __name__ == '__main__':
    # Required for converting emojis to text
    demoji.download_codes()
    ######################################################## These entities need to be added to
    sts_ents = ['@kingjames', '@lakers', '@lakersnation']
    js_ents = ['@potus', '@joebiden']
    
    # Reading data
    sts_ids, sts_labels, sts_texts = read_sts('data/sts_gold_tweet.csv')
    js_ids, js_texts = read_js('data/cleaned_tweets.csv')
    
    # Preprocessing tweets for each dataset
    sts_tokens_pos = preprocess(sts_texts, sts_ents)
    js_tokens_pos = preprocess(js_texts, js_ents)
    
    # Writing to CSVs
    sts_df = pd.DataFrame(list(zip(sts_ids, sts_tokens_pos, sts_labels)),
               columns =['id', 'tokens_pos', 'label'])
    js_df = pd.DataFrame(list(zip(js_ids, js_tokens_pos)),
               columns =['id', 'tokens_pos'])
    sts_df.to_csv('data/sts_tokenized.csv', index=False)
    js_df.to_csv('data/js_tokenized.csv', index=False)
