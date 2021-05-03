import argparse
import pandas as pd


def clean_time(tz, start, end):
    """
    Converts date column to given timezone and filters to tweets written
    between start and end times (inclusive).

    Parameters
    ----------
    tz : str
        Timezone to convert to from UTC
    start : str
        Starting datetime in a form consistent with the datetime you chose
        Ex: For US Eastern, something like '2021-04-28 12:00:00-04:00'
    end : str
        Ending datetime in a form consistent with the datetime you chose

    Returns
    -------
    df : DataFrame
        Converted data

    """
    df = pd.read_pickle('data/scraped_tweets.pkl')
    # Twitter data is originally in UTC
    df['date'] = df['date'].dt.tz_localize('UTC')
    df['date'] = df['date'].dt.tz_convert(tz)
    df = df[(df['date'] >= start) & (df['date'] <= end)].reset_index(drop=True)
    return df


if __name__ == '__main__':
    # Setting up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--timezone', type=str,
                        default='US/Eastern',
                        help="Timezone to convert date column to")
    parser.add_argument('--start_datetime', type=str,
                        default='2021-04-28 12:00:00-04:00',
                        help="Starting datetime for filtering tweets")
    parser.add_argument('--end_datetime', type=str,
                        default='2021-04-29 12:00:00-4:00',
                        help="Ending datetime for filtering tweets")
    args = parser.parse_args()

    # Converting timezone and filtering by start/end datetimes
    data = clean_time(args.timezone, args.start_datetime, args.end_datetime)
    
    
    data.to_csv('data/cleaned_tweets.csv', index=False)


