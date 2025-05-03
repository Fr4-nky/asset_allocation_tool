import base64
import pandas as pd
import requests
import json
import streamlit as st
import time

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
def fetch_and_decode(url, column_name, retries=3, initial_backoff=1):
    """Fetches data from a URL, decodes it, and returns a Pandas DataFrame with retries."""
    print(f"INFO: Fetching data from {url}...")
    backoff = initial_backoff
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=30) 
            response.raise_for_status() # Check for HTTP errors (4xx or 5xx)
            encoded_data = response.json()
            decoded_data = decode_base64_data(encoded_data)
            if not decoded_data:
                print(f"WARNING: No valid data decoded from {url}. Returning None.")
                return None # No point retrying if decoding fails
            df = pd.DataFrame(decoded_data, columns=['Date', column_name])
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            print(f"SUCCESS: Successfully fetched and processed data for {column_name}.")
            return df # Success, exit retry loop
        
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"WARNING: Attempt {attempt + 1}/{retries + 1} failed for {url} due to network error: {e}")
            if attempt < retries:
                print(f"Retrying in {backoff:.2f} seconds...")
                time.sleep(backoff)
                backoff *= 2 # Exponential backoff
            else:
                print(f"ERROR: Max retries reached for {url}. Network error: {e}")
                return None
                
        except requests.exceptions.HTTPError as e:
            # Non-transient HTTP errors (e.g., 404 Not Found, 403 Forbidden)
            print(f"ERROR: HTTP error fetching data from {url}: {e}. No retry needed.")
            return None # Don't retry on client/server errors like 404
            
        except requests.exceptions.RequestException as e:
            # Other request exceptions (e.g., Invalid URL)
            print(f"ERROR: Request error fetching data from {url}: {e}. No retry needed.")
            return None # Don't retry on fundamental request issues
            
        except json.JSONDecodeError as e:
            print(f"ERROR: Error decoding JSON from {url}. Response text: {response.text[:200]}... Error: {e}. No retry needed.")
            return None # Don't retry if content is not JSON
            
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while processing {url}: {e}. No retry needed.")
            return None # Don't retry on unexpected errors
            
    # Should theoretically not be reached if logic is correct, but as a fallback:
    print(f"ERROR: Failed to fetch data from {url} after {retries + 1} attempts.")
    return None
