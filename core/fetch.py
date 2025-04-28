import base64
import pandas as pd
import requests
import json
import streamlit as st

def decode_base64_data(encoded_data):
    """Decodes a list of [base64_date, base64_value] pairs."""
    decoded_list = []
    for date_b64, value_b64 in encoded_data:
        try:
            date_str = base64.b64decode(date_b64).decode('utf-8')
            value_str = base64.b64decode(value_b64).decode('utf-8')
            value_float = float(value_str)
            decoded_list.append([date_str, value_float])
        except (base64.binascii.Error, UnicodeDecodeError, ValueError) as e:
            print(f"WARNING: Skipping record due to decoding/conversion error: {e} - Date: {date_b64}, Value: {value_b64}")
    return decoded_list

@st.cache_data
def fetch_and_decode(url, column_name):
    """Fetches data from a URL, decodes it, and returns a Pandas DataFrame."""
    print(f"INFO: Fetching data from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        encoded_data = response.json()
        decoded_data = decode_base64_data(encoded_data)
        if not decoded_data:
            print(f"WARNING: No valid data decoded from {url}")
            return None
        df = pd.DataFrame(decoded_data, columns=['Date', column_name])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        print(f"SUCCESS: Successfully fetched and processed data for {column_name}.")
        return df
    except requests.exceptions.Timeout:
        print(f"ERROR: Request timed out while fetching data from {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Error fetching data from {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Error decoding JSON from {url}. Response text: {response.text[:500]}... Error: {e}")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while processing {url}: {e}")
        return None
