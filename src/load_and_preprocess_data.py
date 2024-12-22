# Import necessary libraries
# import pandas
import pandas as pd

def load_data(path='../data/green_tripdata_2022-01.parquet'):
    df = pd.read_parquet(path)
    return df

def preprocess_data(df):
    """
    Calculate the duration of each trip in minutes
    This is done by subtracting the pickup time from the dropoff time
    """
    # convert from seconds to minutes
    df["duration_min"] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60

    # Filter out trips with unrealistic durations
    # We only keep trips with durations between 0 and 60 minutes
    df = df[(df.duration_min >= 0) & (df.duration_min <= 60)]

    # Filter out trips with unrealistic passenger counts
    # We only keep trips with passenger counts between 1 and 8
    df = df[(df.passenger_count > 0) & (df.passenger_count <= 8)]

    # Return the preprocessed data
    return df

if __name__ == "__main__":
    jan_data = load_data()
    preprocessed_data = preprocess_data(jan_data)