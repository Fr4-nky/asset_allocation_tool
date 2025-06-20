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
            pass
    return decoded_list

@st.cache_data
def fetch_and_decode(url, column_name, retries=3, initial_backoff=1):
    """Fetches data from a URL, decodes it, and returns a Pandas DataFrame with retries."""

    backoff = initial_backoff
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=30) 
            response.raise_for_status() # Check for HTTP errors (4xx or 5xx)
            encoded_data = response.json()
            decoded_data = decode_base64_data(encoded_data)
            if not decoded_data:

                return None # No point retrying if decoding fails
            df = pd.DataFrame(decoded_data, columns=['Date', column_name])
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')

            return df # Success, exit retry loop
        
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:

            if attempt < retries:

                time.sleep(backoff)
                backoff *= 2 # Exponential backoff
            else:

                return None
                
        except requests.exceptions.HTTPError as e:
            # Non-transient HTTP errors (e.g., 404 Not Found, 403 Forbidden)

            return None # Don't retry on client/server errors like 404
            
        except requests.exceptions.RequestException as e:
            # Other request exceptions (e.g., Invalid URL)

            return None # Don't retry on fundamental request issues
            
        except json.JSONDecodeError as e:

            return None # Don't retry if content is not JSON
            
        except Exception as e:

            return None # Don't retry on unexpected errors
            
    # Should theoretically not be reached if logic is correct, but as a fallback:

    return None
