# Entity-based Sentiment Analysis on Tweets
##### Anderson Monken and Nicole Yoder

Here is a description of all the files in this repository in the order that they should be used.

### 1. Scraping Twitter
The following files located in were used to collect tweets related to President Biden's address to the joint session of Congress on 4/28/2021.

|File                                      | Description         |
|----------------------------------------- | ------------------- |
|`.env`                                    | Stores Twitter keys and is listed in `.gitignore`, so that they remain private <br />**Make sure you add this file yourself if you want to run `scraping_tweets.py`** |
|`scripts/1_scraping_js/search_phrases.txt`| Lists the phrases and/or hashtags (on separate lines) used in the Twitter search |
|`scripts/1_scraping_jsscraping_tweets.py` | Uses Twitter API through the `tweepy` package to collect tweets according to search phrases and date arguments |
|`data/scraped_tweets.pkl`                 | Resulting data file |

Usage of the `scraping_tweets.py` script is as follows:

`python scraping_tweets.py --search_phrases_file 'search_phrases.txt' --date_since '2021-04-28' --date_until '2021-04-30' --num_tweets None`

A couple of notes:
- For the default `date_since` and `date_until`, it would search for tweets during the 28th and the 29th (the until date is not included).
- The `num_tweets` argument allows you to cut off the search after a given number of tweets, but the default `None` results in all of the tweets being collected
- This script takes a long time to run because the Twitter API rate limits it every 2700 or so tweets, and then it sleeps for ~12 minutes before starting again.
- You can use `Ctrl-C` to interrupt the search and still save what data you collected so far.


### 2. Cleaning the Joint Session Tweets
The `scripts/2_cleaning_js/cleaning_js.py` script was used to clean the datetime column by converting it from UTC to US/Eastern and limiting the tweets to noon on 4/28 to noon on 4/29. The resulting data file was `data/js_tweets.csv`.

Usage of the `cleaning_js.py` script is as follows:

`python cleaning_js.py --timezone 'US/Eastern' --start_datetime '2021-04-28 12:00:00-04:00' --end_datetime '2021-04-29 12:00:00-4:00'`


### 3. Preprocessing STS and Joint Session Tweets
The following files (located in `scripts/3_preprocessing_sts_js/` unless otherwise noted) were used to preprocess the tweets from either dataset by:
1. Replacing emojis with text (e.g. '🦉' becomes 'owl')
2. Removing urls
3. Tokenizing using nltk TweetTokenizer
4. Taging POS using nltk pos_tag
5. Removing random usernames (e.g. @user1234) but keeps entity usernames (e.g. @POTUS)
6. Converting POS tag for usernames to NNP
7. Converting tokens to bigram tokens
8. Converting synonym tokens to single form

|File                                               | Description         |
|-------------------------------------------------- | ------------------- |
|`js_usernames.txt`<br>`sts_usernames.txt`          | Lists username entities (e.g. '@POTUS') that should not be removed from tweets |
|`js_bigrams.txt`                                   | Lists entity bigrams (e.g. 'joe,biden') |
|`js_synonyms.txt`                                  | Lists synonyms for each entity in one line (e.g. '@potus' and 'biden') |
|`data/js_tokenized.pkl`<br>`data/sts_tokenized.pkl`| Resulting data files |

Usage of the `preprocessing.py` script is as follows:

For the evaluation dataset:
`python preprocessing.py --data sts --usernames_file sts_usernames.txt --bigrams_file None --synonyms_file None`

For the experimental dataset:
`python preprocessing.py --data js --usernames_file js_usernames.txt --bigrams_file js_bigrams.txt --synonyms_file js_synonyms.txt`


### 4. SentiCircles
The `scripts/3_senticircles/senticircles.py` script was used to ANDERSON WRITE DESCRIPTION. Plots of each entity's senticircle can be found in `plots/js/` or `plots/sts`.

Usage of the `senticircles.py` script is as follows:

ANDERSON FILL OUT USAGE


### Miscellaneous
|File                                           | Description         |
|---------------------------------------------- | ------------------- |
|`.gitignore`                                   | Lists files that should be ignored by git |
|`README.md`                                    | This markdown document |
|`requirements.txt`                             | Lists packages needed to run the scripts |
|`Entity-Based Sentiment Analysis on Tweets.pdf`| PDF of our presentation slides |
|`WHATEVER PDF FILE WE USE FOR OUR PAPER`       | PDF of our paper |
