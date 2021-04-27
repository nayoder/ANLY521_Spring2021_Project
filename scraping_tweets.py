import argparse
import os
from dotenv import load_dotenv
import pandas as pd
import tweepy
import time

load_dotenv()

# Put these variables in a .env file in the parent directory
TWITTER_CONSUMER_KEY = os.getenv('TWITTER_CONSUMER_KEY')
TWITTER_CONSUMER_SECRET = os.getenv('TWITTER_CONSUMER_SECRET')
TWITTER_ACCESS_KEY = os.getenv('TWITTER_ACCESS_KEY')
TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET')


        
def create_query(filename):
    """
    Converts a text file of newline-separated Twitter search phrases into a
    Tweepy API query separated by ORs.

    Parameters
    ----------
    filename : str
        Name of text file with Twitter search phrases.

    Returns
    -------
    query : str
        String in Tweepy query format (separated by ORs).

    """
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if i == 0:
                query = line.strip()
            if i > 0:
                query += f' OR {line.strip()}'
    query += ' -filter:retweets'
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
    ith_tweet : list
        List of the following data retrieved using Tweepy:
            [username, description, location, following,
             followers, totaltweets, retweetcount, text, hashtext]

    Returns
    -------
    None.

    """
    
    print(f"""
Tweet {i}:
Username:{ith_tweet[0]}
Date:{ith_tweet[1]}
Description:{ith_tweet[2]}
Location:{ith_tweet[3]}
Following Count:{ith_tweet[4]}
Follower Count:{ith_tweet[5]}
Total Tweets:{ith_tweet[6]}
Retweet Count:{ith_tweet[7]}
Tweet Text:{ith_tweet[8]}
Hashtags Used:{ith_tweet[9]}
""")


    
def limit_handled(cursor):
    """
    Yields the next tweet retrieved by the cursor, but handles the RateLimitError
    and StopIteration at the end of the generator.

    Parameters
    ----------
    cursor : tweepy.cursor.ItemIterator
        tweepy.Cursor used to search for tweets

    Yields
    ------
    tweepy.models.Status
        tweet data

    """
    while True:
        try:
            yield next(cursor)
        except tweepy.RateLimitError:
            print('Rate Limit Error, waiting 15 minutes')
            time.sleep(15 * 60)
        except StopIteration:
            return

  

def scrape(phrases_filename, date_since, numtweet):
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
        Date in the form 'yyyy-mm-dd' to search since.
    numtweet : int
        Number of tweets to limit to if wanted.

    Returns
    -------
    None.

    """
    # Creating Tweepy query version of search phrases
    search_phrases = create_query(phrases_filename)

      
    # Creating DataFrame using pandas
    db = pd.DataFrame(columns=['username', 'date', 'description', 'location', 'following',
                               'followers', 'totaltweets', 'retweetcount', 'text', 'hashtags'])
      
    # We are using .Cursor() to search through twitter for the required tweets.
    # The number of tweets can be restricted using .items(number of tweets)
    if numtweet == None:
        tweets = tweepy.Cursor(api.search, q=search_phrases, lang="en",
                               since=date_since, tweet_mode='extended').items()
    else:
        tweets = tweepy.Cursor(api.search, q=search_phrases, lang="en",
                               since=date_since, tweet_mode='extended').items(numtweet)
     
      
    # Counter to maintain Tweet Count
    i = 1  
      
    # We will iterate over each tweet in the list to extract information about each tweet
    for tweet in limit_handled(tweets):
        username = tweet.user.screen_name
        date = tweet.created_at
        description = tweet.user.description
        location = tweet.user.location
        following = tweet.user.friends_count
        followers = tweet.user.followers_count
        totaltweets = tweet.user.statuses_count
        retweetcount = tweet.retweet_count
        text = tweet.full_text
        hashtags = tweet.entities['hashtags']
          
        # Getting hashtags into a list
        hashtext = list()
        for j in range(0, len(hashtags)):
            hashtext.append(hashtags[j]['text'])
          
        # Here we are appending all the extracted information in the DataFrame
        ith_tweet = [username, date, description, location, following,
                     followers, totaltweets, retweetcount, text, hashtext]
        db.loc[len(db)] = ith_tweet
          
        # Printing tweet data for every 10th tweet
        if i % 10 == 0:
            print_tweet(i, ith_tweet)
        i = i+1
    
      
    # We will save our database as a CSV file.
    filename = 'scraped_tweets.csv'
    db.to_csv(filename, index=False)

    

  
if __name__ == '__main__':
    
    # Setting up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_phrases_file", type=str,
                        default="search_phrases.txt",
                        help="Text file of the hashtags/phrases you want to search for")
    parser.add_argument("--date", type=str,
                        default="2021-04-26",######################################### CHANGE TO THE 28th
                        help="Starting date for tweet search")
    parser.add_argument("--num_tweets", type=int,
                        default=None,################################################# possibly change
                        help="Number of tweets to search for")
    args = parser.parse_args()


    # Enter your own credentials obtained from your Twitter developer account
    # into the .env file
    auth = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
    auth.set_access_token(TWITTER_ACCESS_KEY, TWITTER_ACCESS_SECRET)
    api = tweepy.API(auth)
    
    scrape(args.search_phrases_file, args.date, args.num_tweets)
    print('Scraping has completed!')