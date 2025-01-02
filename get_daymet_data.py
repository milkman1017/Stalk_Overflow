import requests
import pandas as pd
from datetime import datetime
from io import StringIO
from tqdm import tqdm

# Function to fetch Daymet data for specific days within a year

def fetch_daymet_by_days(lat, lon, year, days):
    """
    Fetch Daymet data for a specific latitude, longitude, year, and specific days.
    """
    url = "https://daymet.ornl.gov/single-pixel/api/data"
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    params = {
        "lat": lat,
        "lon": lon,
        "vars": "T2MWET,QV2M,RH2M,T2M_MAX,ALLSKY_SFC_SW_DWN,PS,T2MDEW,WS2M,T2M_MIN,T2M,PRECTOTCORR",
        "start": start_date,
        "end": end_date,
        "format": "csv"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        daymet_data = pd.read_csv(StringIO(response.text), skiprows=6)  # Adjust skiprows for header lines
        # print(daymet_data[daymet_data['yday'].isin(days)])
        return daymet_data[daymet_data['yday'].isin(days)]
    else:
        print(f"Error fetching data for {lat}, {lon} in year {year}: {response.status_code}")
        return None

def create_daymet_dataframe(weather_data, metadata):
    """
    Merge weather_data with metadata, fetch Daymet data by specific days, and return a new DataFrame.
    """
    # Ensure Env is the index for metadata for easy mapping
    if "Env" not in metadata.index:
        metadata.set_index("Env", inplace=True)

    # Convert Date column to string
    weather_data["Date"] = weather_data["Date"].astype(str)

    # Cache for storing fetched Daymet data by (lat, lon, year)
    daymet_cache = {}

    # Initialize list for storing results
    results = []

    # Group weather_data by environment and year
    for (env, year), group in tqdm(weather_data.groupby(["Env", weather_data["Date"].str[:4].astype(int)])):
        try:
            # Get latitude and longitude for the current environment
            lat = metadata.loc[env, "Weather_Station_Latitude (in decimal numbers NOT DMS)"]
            lon = metadata.loc[env, "Weather_Station_Longitude (in decimal numbers NOT DMS)"]
        except KeyError:
            print(f"Environment '{env}' not found in metadata.")
            continue

        # Extract unique days of the year for the current group
        days = group["Date"].apply(lambda x: datetime.strptime(x, "%Y%m%d").timetuple().tm_yday).unique()

        # Check if data for this (lat, lon, year) is already fetched
        cache_key = (lat, lon, year)
        if cache_key not in daymet_cache:
            daymet_data = fetch_daymet_by_days(lat, lon, year, days)
            if daymet_data is not None:
                daymet_cache[cache_key] = daymet_data
            else:
                continue
        else:
            daymet_data = daymet_cache[cache_key]

        # Add environment and date columns to the filtered Daymet data
        for _, row in group.iterrows():
            date_str = row["Date"]
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            yday = date_obj.timetuple().tm_yday

            # Filter Daymet data for the specific day
            day_data = daymet_data[daymet_data['yday'] == yday]
            if not day_data.empty:
                day_data = day_data.copy()
                day_data["Env"] = env
                day_data["Date"] = date_str
                results.append(day_data)
                # print(results)

    # Combine all results into a single DataFrame
    if results:
        final_df = pd.concat(results, ignore_index=True)
        # Set index to Env and Date
        final_df.set_index(["Env", "Date"], inplace=True)
        return final_df
    else:
        print("No Daymet data matched the input criteria.")
        return pd.DataFrame()


# Load the required data files
weather_data = pd.read_csv("Training_data/4_Training_Weather_Data_2014_2023_seasons_only.csv")  # Replace with your actual file path
metadata = pd.read_csv("Training_data/2_Training_Meta_Data_2014_2023.csv")  # Replace with your actual file path

# Create the final Daymet DataFrame
daymet_df = create_daymet_dataframe(weather_data, metadata)

# Save the resulting Daymet data to a CSV file
output_file = "daymet_data_filtered.csv"
daymet_df.to_csv(output_file)
print(f"Daymet data saved to '{output_file}'")
