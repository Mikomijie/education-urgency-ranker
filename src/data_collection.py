"""
Data Collection Module
Fetches education data from EMIS portal for all Nigerian LGAs
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import json

def fetch_emis_data():
    """
    Fetch LGA-level education data from EMIS portal
    Returns: DataFrame with enrollment, teachers, facilities by LGA
    """
    # EMIS API endpoint (public data)
    url = "https://emis.education.gov.ng/portal/"
    
    print("Fetching EMIS data...")
    try:
        response = requests.get(url, timeout=10)
        print(f"✅ Connected to EMIS portal")
    except:
        print("⚠️ EMIS portal unreachable - using fallback data")
        return load_fallback_data()
    
    # For now, return placeholder structure
    # We'll populate this with real scraping in next step
    data = {
        'LGA': [],
        'State': [],
        'Enrollment_2024': [],
        'Teachers_2024': [],
        'Classrooms': [],
        'Toilets': []
    }
    
    return pd.DataFrame(data)

def load_fallback_data():
    """Fallback: Load from cached/synthetic data if API unavailable"""
    print("Loading fallback dataset...")
    # Placeholder - we'll get real data next
    return pd.DataFrame()

def clean_data(df):
    """Clean and validate education data"""
    df = df.dropna(subset=['Enrollment_2024', 'Teachers_2024'])
    df['StudentTeacherRatio'] = df['Enrollment_2024'] / (df['Teachers_2024'] + 1)
    return df

def save_data(df, path='data/emis_lga.csv'):
    """Save processed data to CSV"""
    df.to_csv(path, index=False)
    print(f"✅ Data saved to {path}")

if __name__ == "__main__":
    df = fetch_emis_data()
    df = clean_data(df)
    save_data(df)
    print(f"Dataset shape: {df.shape}")