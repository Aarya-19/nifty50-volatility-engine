import requests
import pandas as pd
import zipfile
import io
from datetime import datetime, timedelta

# Configuration for your request
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"  # Adjust as needed
STRIKE_COUNT = 30         # 15 ITM + 15 OTM (approx)
EXPIRY_COUNT = 6          # Nearest 6 weekly expiries

# NSE Headers to avoid 403 Forbidden errors
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/all-reports-derivatives"
}

def get_nse_url(date_obj):
    """
    Determines the correct URL based on the format transition date (June 2024).
    """
    # Format transition date
    transition_date = datetime(2024, 6, 21)
    
    date_str = date_obj.strftime("%Y%m%d")
    day = date_obj.strftime("%d")
    month_abbr = date_obj.strftime("%b").upper()
    year = date_obj.strftime("%Y")

    if date_obj < transition_date:
        # Legacy format: foDDMMMYYYYbhav.csv.zip
        return f"https://nsearchives.nseindia.com{year}/{month_abbr}/fo{day}{month_abbr}{year}bhav.csv.zip", "legacy"
    else:
        # New UDiFF format: BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip
        return f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv.zip", "udiff"

def process_data(df, format_type):
    """
    Standardizes column names and filters for Nifty 50 options.
    """
    if format_type == "legacy":
        # Legacy Columns: INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP, CLOSE, etc.
        df = df[df['SYMBOL'] == 'NIFTY']
        df = df[df['INSTRUMENT'].str.contains('OPTIDX', na=False)]
        
        # Mapping to user requirements
        df = df.rename(columns={
            'TIMESTAMP': 'Date',
            'EXPIRY_DT': 'Expiry',
            'STRIKE_PR': 'Strike',
            'OPTION_TYP': 'Type',
            'CLOSE': 'Price'
        })
    else:
        # UDiFF Columns: TradDt, TckrSymb, XpryDt, StrkPric, OptnTp, ClsPric, etc.
        df = df[df['TckrSymb'] == 'NIFTY']
        df = df[df['FinInstrmTp'] == 'IDO'] # IDO stands for Index Options
        
        # Mapping to user requirements
        df = df.rename(columns={
            'TradDt': 'Date',
            'XpryDt': 'Expiry',
            'StrkPric': 'Strike',
            'OptnTp': 'Type',
            'ClsPric': 'Price'
        })
        
    return df[['Date', 'Expiry', 'Strike', 'Type', 'Price']]

def download_bhavcopy(date_obj):
    """
    Downloads, extracts, and parses the daily Bhavcopy.
    """
    url, format_type = get_nse_url(date_obj)
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Extract the single CSV from the zip
                filename = z.namelist()[0]
                with z.open(filename) as f:
                    return pd.read_csv(f), format_type
    except Exception as e:
        print(f"Skipping {date_obj.date()}: Market likely closed or {e}")
    return None, None

def run_retriever():
    all_dates = pd.date_range(START_DATE, END_DATE)
    master_list = []

    print(f"Starting retrieval from {START_DATE} to {END_DATE}...")

    for current_date in all_dates:
        raw_df, fmt = download_bhavcopy(current_date)
        
        if raw_df is not None:
            # 1. Filter Nifty options
            nifty_opt = process_data(raw_df, fmt)
            
            if not nifty_opt.empty:
                # 2. Get next 6 weekly expiries
                unique_expiries = sorted(pd.to_datetime(nifty_opt['Expiry']).unique())
                target_expiries = unique_expiries[:EXPIRY_COUNT]
                nifty_opt = nifty_opt[pd.to_datetime(nifty_opt['Expiry']).isin(target_expiries)]
                
                # 3. Calculate ATM Strike
                # Logic: Find the strike where CE and PE prices are closest to each other
                # (Simple proxy for current spot in the absence of separate spot file)
                unique_strikes = sorted(nifty_opt['Strike'].unique())
                
                # Filter for ~30 strikes around the median available strike
                mid_idx = len(unique_strikes) // 2
                start_idx = max(0, mid_idx - 15)
                end_idx = min(len(unique_strikes), mid_idx + 15)
                target_strikes = unique_strikes[start_idx:end_idx]
                
                nifty_opt = nifty_opt[nifty_opt['Strike'].isin(target_strikes)]
                
                master_list.append(nifty_opt)
                print(f"Processed: {current_date.date()} | Found {len(nifty_opt)} contracts.")

    if master_list:
        final_df = pd.concat(master_list)
        output_file = "Nifty_Options_Historical_2022_2025.csv"
        final_df.to_csv(output_file, index=False)
        print(f"\nSUCCESS: Data saved to {output_file}")
    else:
        print("\nNo data retrieved. Check headers or internet connection.")

if __name__ == "__main__":
    run_retriever()
