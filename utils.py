import pandas as pd


def read_data(file_path: str) -> pd.DataFrame:
    """
    Reads the processed NetSuite sales data and returns a DataFrame.

    Returns:
    - df (pd.DataFrame): The processed NetSuite sales data.
    """
    # Read the processed NetSuite sales data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError("The processed NetSuite sales data file was not found. Please ensure that the file exists.")
    try:
        # Drop columns that are not needed
        not_needed_columns = ['Inventory_Item_NameNumber', 'Transactions_Net_Sales_For_Product_', 'Transactions_Transaction_Number']
        df.drop(not_needed_columns, axis=1, inplace=True)

        # Rename the Date column
        df.rename(columns={'Date_Transaction_Date': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])

        # Step 1: Sort the DataFrame by 'Date' (ascending order)
        df = df.sort_values(by='Date')
        # print the first few rows of the Date column
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        return df
    except Exception as e:
        raise Exception(f"An error occurred during data processing: {str(e)}")


def get_valid_categories() -> list:
    return ['Other', 'Dresses', 'Home Decor', 'Denim', 'Sweaters', 'Pants',
       'Blouses', 'Jewellery', 'T-Shirts', 'Skirts', 'Holiday', 'Gifts',
       'Accessories', 'Jackets', 'Outerwear', 'Health & Beauty',
       'Food & Drink', 'Sleepwear', 'Bags', 'Intimates', 'Shorts',
       'Bodysuits', 'Footwear', 'Stationary', 'Jumpsuits', 'Furniture',
       'Unknown', 'Rompers', 'Props']