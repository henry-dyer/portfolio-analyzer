import requests
from http import HTTPStatus
import logging
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta


KEY = 'jfwIWcXLBl_Mz30S3sHudfG9tGmVUPxZ'


def fetch_polygon_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    timespan: str = 'day',
    multiplier: str = '1'
) -> pd.DataFrame:
    """
    Fetch aggregated price data from Polygon.io API.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date for data retrieval
    end_date : str
        End date for data retrieval
    timespan : str, optional
        Time interval for aggregation (default: 'day')
    multiplier : str, optional
        Number of timespans to aggregate (default: '1')

    Returns
    -------
    pd.DataFrame
        DataFrame containing price and volume data

    Raises
    ------
    requests.RequestException
        If API request fails
    ValueError
        If API returns invalid data
    """
    API_KEY = KEY
    api_endpoint = (
        f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/'
        f'{multiplier}/{timespan}/{start_date}/{end_date}?apiKey={API_KEY}'
    )

    try:
        response = requests.get(api_endpoint, timeout=10)
        
        if response.status_code == HTTPStatus.OK:
            response_data = response.json()
            
            if not response_data.get('results'):
                raise ValueError(f"No data returned for ticker {ticker}")
                
            price_df = pd.DataFrame.from_dict(response_data['results'])
            
            # Rename columns to be more descriptive
            column_mapping = {
                'v': 'volume',
                'o': 'price_open',
                'c': 'price_close',
                't': 'timestamp',
                'h': 'price_high',
                'l': 'price_low',
                'n': 'num_trades',
                'vw': 'volume_weighted_price'
            }
            
            price_df = price_df.rename(columns=column_mapping)
            
            # Add metadata
            price_df['ticker'] = ticker
            price_df['timestamp'] = pd.to_datetime(price_df.timestamp, unit='ms')
            
            return price_df
            
        else:
            error_msg = f"API request failed with status code: {response.status_code}"
            if response.text:
                error_msg += f", Response: {response.text}"
            raise requests.RequestException(error_msg)
            
    except requests.RequestException as e:
        logging.error(f"API connection error for ticker {ticker}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error processing ticker {ticker}: {str(e)}")
        raise


def process_price_series(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw price data into a clean format with daily prices.

    Parameters
    ----------
    price_df : pd.DataFrame
        Raw price data DataFrame from fetch_polygon_data

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with trading date and price columns

    Raises
    ------
    ValueError
        If input DataFrame is empty or missing required columns
    """
    required_columns = ['timestamp', 'volume_weighted_price', 'ticker']
    missing_columns = [col for col in required_columns if col not in price_df.columns]
    
    if missing_columns:
        raise ValueError(
            f"Input DataFrame missing required columns: {missing_columns}"
        )
    
    if price_df.empty:
        raise ValueError("Input DataFrame is empty")

    # Extract ticker symbol from first row
    ticker_symbol = price_df.loc[0, 'ticker']
    
    # Select and rename columns
    processed_df = price_df[['timestamp', 'volume_weighted_price']].copy()
    processed_df = processed_df.rename(columns={
        'timestamp': 'trading_date',
        'volume_weighted_price': f'price_{ticker_symbol}'
    })
    
    # Convert timestamp to date
    processed_df['trading_date'] = processed_df['trading_date'].dt.date
    
    return processed_df


def pull_return_data(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Pull and merge price data for given tickers.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols to pull data for
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format. Defaults to 1 year ago
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. Defaults to today

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing data for all tickers

    Raises
    ------
    ValueError
        If tickers list is empty or dates are invalid
    """

    # Input validation
    if not tickers:
        raise ValueError("Tickers list cannot be empty")
    

    # Set default dates if not provided
    current_date = datetime.now()
    default_end_date = current_date.strftime('%Y-%m-%d')
    default_start_date = (current_date - timedelta(days=365)).strftime('%Y-%m-%d')
    
    analysis_end_date = end_date or default_end_date
    analysis_start_date = start_date or default_start_date

    # Validate dates
    try:
        pd.to_datetime([analysis_start_date, analysis_end_date])
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")
    
    if pd.to_datetime(analysis_start_date) > pd.to_datetime(analysis_end_date):
        raise ValueError("Start date cannot be after end date")

    merged_data = None
    for symbol in tickers:
        try:
            # Pull and clean data for each ticker
            raw_price_data = fetch_polygon_price_data(
                ticker=symbol, 
                start_date=analysis_start_date, 
                end_date=analysis_end_date
            )
            cleaned_price_data = process_price_series(raw_price_data)
            
            # Initialize or merge with output
            if merged_data is None:
                merged_data = cleaned_price_data
            else:
                merged_data = merged_data.merge(
                    cleaned_price_data, 
                    on='trading_date', 
                    how='outer'
                )
                
        except Exception as e:
            logging.error(f"Error processing ticker symbol {symbol}: {e}")
            continue
    
    if merged_data is None:
        raise ValueError("No valid data retrieved for any ticker symbol")

    sorted_data = merged_data.sort_values('trading_date')
    
    return sorted_data

def sample() -> None:
    start_date = '2023-01-01'
    end_date = '2024-01'

    tickers = ['AAPL', 'TSLA', 'JPM', 'NVDA']

    data = pull_return_data(tickers, start_date=start_date, end_date=end_date)

    print(data.head())

    data.to_csv('sample.csv', index=False)

    return

def main() -> None:

    pass


if __name__ == '__main__':
    main()