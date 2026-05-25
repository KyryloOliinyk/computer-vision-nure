import argparse
import json
from pathlib import Path

import pandas as pd
import requests


def fetch_exchange_data(currency_code: str, start_date: str, end_date: str):
    """
    Fetches exchange rate data from the NBU API.
    """
    url = (
        f"https://bank.gov.ua/NBU_Exchange/exchange_site"
        f"?valcode={currency_code.upper()}&start={start_date}&end={end_date}&sort=exchangedate&order=desc&json"
    )
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def save_to_csv(data: json, output_file: Path):
    """
    Saves exchange rate data to a CSV file.
    """
    df = pd.DataFrame(data)
    df = df[['exchangedate', 'rate']]
    df.columns = ['Date', 'Rate']
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    df = df.sort_values('Date')
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} rows to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Download exchange rate data from NBU API and save as CSV.")
    parser.add_argument('--currency_code', type=str, required=True, help='Currency code (e.g. USD, EUR)')
    parser.add_argument('--start', type=str, required=True, help='Start date in YYYYMMDD format')
    parser.add_argument('--end', type=str, required=True, help='End date in YYYYMMDD format')
    parser.add_argument('--output', type=Path, required=True, help='Path to output CSV file')

    args = parser.parse_args()
    data = fetch_exchange_data(args.currency_code, args.start, args.end)
    print("Data:", json.dumps(data, indent=4, ensure_ascii=False))
    save_to_csv(data, args.output)


if __name__ == "__main__":
    main()
