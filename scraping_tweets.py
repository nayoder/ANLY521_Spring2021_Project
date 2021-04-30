import argparse
import os
from dotenv import load_dotenv
import pandas as pd
import tweepy


load_dotenv()

# Put these variables in a .env file in the parent directory
TWITTER_CONSUMER_KEY = os.getenv('TWITTER_CONSUMER_KEY')
TWITTER_CONSUMER_SECRET = os.getenv('TWITTER_CONSUMER_SECRET')
TWITTER_ACCESS_KEY = os.getenv('TWITTER_ACCESS_KEY')
TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET')


def create_query(filename):
    """
    Converts a text file of newline-separated Twitter search phrases into a
    Tweepy API query separated by "OR"s.

    Parameters
    ----------
    filename : str
        Name of text file with Twitter search phrases.

    Returns
    -------
    query : str
        String in Tweepy query format (separated by "OR"s).

    """
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if i == 0:
                query = line.strip()
            if i > 0:
                query += f" OR {line.strip()}"
    query += " -filter:retweets"
    return query




def print_tweet(i, ith_tweet):
    """
    Prints the tweet number, username, description, location, following count,
    follower count, total tweets, retweet count, tweet text, and hashtags used
    for the ith tweet found.

    Parameters
    ----------
    i : int
        Number of tweet you want to print.
    ith_tweet : dictionary
        Dictionary of the following data retrieved using Tweepy:
        username, description, location, following,
        followers, totaltweets, retweetcount, text, hashtext

    Returns
    -------
    None.

    """

    print(f"""
Tweet {i}:
Username:{ith_tweet["username"]}
Date:{ith_tweet["date"]}
Description:{ith_tweet["description"]}
Location:{ith_tweet["location"]}
Following Count:{ith_tweet["following"]}
Follower Count:{ith_tweet["followers"]}
Total Tweets:{ith_tweet["totaltweets"]}
Retweet Count:{ith_tweet["retweetcount"]}
Tweet Text:{ith_tweet["text"]}
Hashtags Used:{ith_tweet["hashtext"]}
""")



def yield_next(cursor):
    """
    Yields the next tweet retrieved by the cursor, but handles the StopIteration
    at the end of the generator and allows the user to interrupt with Ctrl-C.

    Parameters
    ----------
    cursor : tweepy.cursor.ItemIterator
        tweepy.Cursor used to search for tweets

    Yields
    ------
    tweepy.models.Status
        tweet data

    """
    try:
        while True:
            try:
                yield next(cursor)
            except StopIteration:
                return
    except KeyboardInterrupt:
        print("\nScraping tweets has been interrupted!")



def clean_write(df, filename):
    """
    Converts following, followers, totaltweets, and retweetcount columns to int,
    and then writes the dataframe to a pickle file.

    Parameters
    ----------
    df : DataFrame
        DataFrame created by scraping tweets.
    filename : str
        Name of output file.

    Returns
    -------
    None.

    """
    print('\n\n--------------------------------------------------------------------------------')
    print('Printing dataframe details:')
    print(df.head())
    print(df.info())
    
    # Certain columns need to be converted to int
    cols = ['following', 'followers', 'totaltweets', 'retweetcount']
    df[cols] = df[cols].astype('int')
    print('\n\nConverted int columns:')
    print(df.info())
    
    # Writing to pickle file
    path = 'data/' + filename
    df.to_pickle(path)


def scrape(phrases_filename, date_since, date_until, numtweet):
    """
    Uses Tweepy API to search for tweets since a given date that use given
    particular phrases and then creates a csv file with the following data:
        username, date, description, location, following, followers,
        totaltweets, retweetcount, text, hashtags

    Parameters
    ----------
    phrases_filename : str
        Name of file that has search phrases/hashtags.
    date_since : str
        Date in the form 'yyyy-mm-dd' to search since (inclusive).
    date_until : str
        Date in the form 'yyyy-mm-dd' to search until (non-inclusive).
        Ex: If date_since='2021-04-28' and date_until='2021-04-30', then the 
        28th and the 29th will be included--not the 30th.
    numtweet : int
        Number of tweets to limit to if wanted.

    Returns
    -------
    None.

    """
    # Creating Tweepy query version of search phrases
    search_phrases = create_query(phrases_filename)
    print("Searching for tweets using the following query:")
    print(search_phrases)


    # Creating DataFrame using pandas
    df = pd.DataFrame(columns=['username', 'date', 'description', 'location', 'following',
                               'followers', 'totaltweets', 'retweetcount', 'text', 'hashtags'])

    # We are using .Cursor() to search through twitter for the required tweets.
    # The number of tweets can be restricted using .items(number of tweets)
    if numtweet is None:
        tweets = tweepy.Cursor(api.search, q=search_phrases, lang='en',
                               since=date_since, until=date_until,
                               tweet_mode='extended').items()
    else:
        tweets = tweepy.Cursor(api.search, q=search_phrases, lang='en',
                               since=date_since, until=date_until,
                               tweet_mode='extended').items(numtweet)

    # Counter to maintain Tweet Count
    i = 1
    
    # Iterate over each tweet in the list to extract information about each tweet
    for tweet in yield_next(tweets):
        # Getting hashtags into a list
        hashtags = tweet.entities['hashtags']
        hashtext = list()
        for j in range(0, len(hashtags)):
            hashtext.append(hashtags[j]['text'])
        
        tweet_data = {
            'username' : tweet.user.screen_name,
            'date' : tweet.created_at,
            'description' : tweet.user.description,
            'location' : tweet.user.location,
            'following' : tweet.user.friends_count,
            'followers' : tweet.user.followers_count,
            'totaltweets' : tweet.user.statuses_count,
            'retweetcount' : tweet.retweet_count,
            'text' : tweet.full_text,
            'hashtext' : hashtext
        }
        
        # Here we are appending all the extracted information in the DataFrame
        df = df.append(tweet_data, ignore_index=True)
        
        # Printing tweet data for every 10th tweet
        if i % 10 == 0:
            print_tweet(i, tweet_data)
        i = i+1

    
    # Saving our dataframe as a pickle file
    clean_write(df, 'scraped_tweets.pkl')




if __name__ == '__main__':
    # Setting up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_phrases_file', type=str,
                        default='search_phrases.txt',
                        help="Text file of the hashtags/phrases you want to search for")
    parser.add_argument('--date_since', type=str,
                        default='2021-04-28',
                        help="Starting date for tweet search")
    parser.add_argument('--date_until', type=str,
                        default='2021-04-30',
                        help="Starting date for tweet search")
    parser.add_argument('--num_tweets', type=int,
                        default=None,
                        help="Number of tweets to search for")
    args = parser.parse_args()


    # Enter your own credentials obtained from your Twitter developer account
    # into the .env file
    auth = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
    auth.set_access_token(TWITTER_ACCESS_KEY, TWITTER_ACCESS_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    
    scrape(args.search_phrases_file, args.date_since, args.date_until, args.num_tweets)
