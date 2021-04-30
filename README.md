# Entity-based Sentiment Analysis on Tweets
##### Anderson Monken and Nicole Yoder

### Scraping Twitter
The following files were used to collect tweets related to President Biden's address to the joint session of Congress on 4/28/2021.

|File                     | Description   |
|------------------------ | ------------- |
|`.env`                   | Stores Twitter keys and is listed in `.gitignore`, so that they remain private <br />**Make sure you add this file yourself if you want to run `scraping_tweets.py`** |
|`search_phrases.txt`     | Lists the phrases and/or hashtags (on separate lines) used in the Twitter search |
|`scraping_tweets.py`     | Uses Twitter API through the `tweepy` package to collect tweets according to search phrases and date arguments |
|`data/scraped_tweets.pkl`| Resulting data file |

The usage of the `scraping_tweets.py` script is as follows:

`python scraping_tweets.py --search_phrases_file 'search_phrases.txt' --date_since '2021-04-28' --date_until '2021-04-30' --num_tweets None`

A couple of notes:
- For the default `date_since` and `date_until`, it would search for tweets during the 28th and the 29th (the until date is not included).
- The `num_tweets` argument allows you to cut off the search after a given number of tweets, but the default `None` results in all of the tweets being collected
- This script takes a long time to run because the Twitter API rate limits it every 2700 or so tweets, and then it sleeps for ~12 minutes before starting again.
- You can use `Ctrl-C` to interrupt the search and still save what data you collected so far.
