import pandas as pd

# Specify the path to your CSV file
csv_file_path = 'gaming_100mb.csv'

    # Read the CSV file into a DataFrame
    # 'sep' argument specifies the delimiter (default is comma)
df = pd.read_csv(csv_file_path, sep=',')

    # Display the first few rows of the DataFrame to verify
print(df.info())