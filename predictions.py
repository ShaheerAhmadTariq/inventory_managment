import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from datetime import timedelta, date

features = [
    'Inventory_Product_Category', 'Inventory_Location_Abbreviation', 'Inventory_Themes',
    'Inventory_MH3_Class', 'Inventory_MH1_Division', 'Inventory_Lifestyle_Category',
    'Inventory_Size', 'Inventory_Base_Price_', 'Inventory_Original_Price_',
    'DayOfWeek', 'IsWeekend', 'Christmas', 'Halloween', 'New Year', 'Independence Day',
    'Easter', 'IsHoliday', 'MeanSales', 'MedianSales', 'Month', 'Quarter', 'WeekOfYear'
]
target = 'Transactions_Units_Sold_Net_Qty'

def inference(df: pd.DataFrame, product_category: str, time_interval: int = 60) -> pd.DataFrame:
    """
    Prepares the dataset by filtering based on Inventory_Product_Category,
    generating dummy data for the next 60 days for each SKU, calculating
    date-related features, and making sales predictions using a pre-trained
    CatBoostRegressor model.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing inventory and sales data.
    - product_category (str): The Inventory_Product_Category to filter the DataFrame.

    Returns:
    - summarize df (pd.DataFrame): A df where each row corresponds
      to a SKU with predictions for the next time_interval days.
    """

    # Helper function to calculate Easter Sunday for a given year
    def calculate_easter(year):
        """Computes Easter Sunday for a given year using Anonymous Gregorian algorithm."""
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f +1) // 3
        h = (19*a + b - d - g +15) % 30
        i = c //4
        k = c %4
        l = (32 + 2*e + 2*i -h -k) %7
        m = (a + 11*h +22*l) //451
        month = (h + l -7*m +114) //31
        day = ((h + l -7*m +114) %31) +1
        return date(year, month, day)

    # Step 1: Filter the DataFrame by Inventory_Product_Category
    filtered_df = df[df['Inventory_Product_Category'] == product_category].copy()

    # Step 2: Identify unique SKUs
    sku_columns = ['Inventory_Item_Internal_ID', 'Inventory_Location_Abbreviation', 'Inventory_Size']
    unique_skus = filtered_df[sku_columns].drop_duplicates()

    # Initialize a list to store summary dictionaries for each SKU
    summary_list = []

    # Iterate over each unique SKU
    for idx, sku in unique_skus.iterrows():
        # Create a filter for the current SKU
        sku_filter = (
            (filtered_df['Inventory_Item_Internal_ID'] == sku['Inventory_Item_Internal_ID']) &
            (filtered_df['Inventory_Location_Abbreviation'] == sku['Inventory_Location_Abbreviation']) &
            (filtered_df['Inventory_Size'] == sku['Inventory_Size'])
        )

        # Extract the DataFrame for the current SKU and sort by Date
        sku_df = filtered_df[sku_filter].sort_values('Date').copy()

        # Step 3: Generate dummy data for the next time_interval days
        last_date = sku_df['Date'].max()
        new_dates = pd.date_range(start=last_date + timedelta(days=1), periods=time_interval, freq='D')
        new_data = pd.DataFrame({'Date': new_dates})

        # Step 4: Calculate date-related features
        new_data['DayOfWeek'] = new_data['Date'].dt.dayofweek
        new_data['IsWeekend'] = new_data['DayOfWeek'].apply(lambda x: 1 if x >=5 else 0)
        new_data['Month'] = new_data['Date'].dt.month
        new_data['Quarter'] = new_data['Date'].dt.quarter
        new_data['WeekOfYear'] = new_data['Date'].dt.isocalendar().week.astype('UInt32')

        # Calculate Easter dates for the years in new_dates
        years = new_data['Date'].dt.year.unique()
        easter_dates = set()
        for yr in years:
            easter_dates.add(calculate_easter(yr))

        # Assign Easter flag
        new_data['Easter'] = new_data['Date'].apply(lambda x: 1 if x.date() in easter_dates else 0)

        # Assign other holiday flags
        new_data['Christmas'] = new_data['Date'].apply(lambda x: 1 if (x.month ==12 and x.day ==25) else 0)
        new_data['Halloween'] = new_data['Date'].apply(lambda x: 1 if (x.month ==10 and x.day ==31) else 0)
        new_data['New Year'] = new_data['Date'].apply(lambda x: 1 if (x.month ==1 and x.day ==1) else 0)
        new_data['Independence Day'] = new_data['Date'].apply(lambda x: 1 if (x.month ==7 and x.day ==4) else 0)

        # IsHoliday flag as any of the specific holidays
        new_data['IsHoliday'] = new_data[['Christmas', 'Halloween', 'New Year', 'Independence Day', 'Easter']].max(axis=1)

        # Step 5: Retain the last known values for other features
        # Identify non-date and non-target columns
        date_related = [
            'Date', 'DayOfWeek', 'IsWeekend', 'Month', 'Quarter', 'WeekOfYear',
            'Christmas', 'Halloween', 'New Year', 'Independence Day', 'Easter', 'IsHoliday'
        ]
        target_column = 'Transactions_Units_Sold_Net_Qty'
        other_features = [col for col in sku_df.columns if col not in date_related + [target_column]]

        # Get the last row to retrieve the last known values
        last_row = sku_df.iloc[-1]

        # Assign the last known values to the new_data
        for col in other_features:
            new_data[col] = last_row[col]

        X_new = new_data[features]

        # Step 6: Load the CatBoost model and make predictions
        model = CatBoostRegressor()
        model.load_model('./models/catboost_model_2.4.cbm')
        predictions = model.predict(X_new)
        # Sum the predictions for the first 7, 15, 30, and 60 days
        if time_interval >= 7:
            sum_7 = predictions[:7].sum()
        if time_interval >= 15:
            sum_15 = predictions[:15].sum()
        if time_interval >= 30:
            sum_30 = predictions[:30].sum()
        if time_interval >= 60:
            sum_60 = predictions[:60].sum()

        # create a summary
        summary = {
            'Inventory_Item_Internal_ID': sku['Inventory_Item_Internal_ID'],
            'Inventory_Location_Abbreviation': sku['Inventory_Location_Abbreviation'],
            'Inventory_Size': sku['Inventory_Size'],
            'Inventory_Product_Category': product_category,
            'Predicted_Sales_7': int(sum_7) if time_interval >= 7 else np.nan,
            'Predicted_Sales_15': int(sum_15) if time_interval >= 15 else np.nan,
            'Predicted_Sales_30': int(sum_30) if time_interval >= 30 else np.nan,
            'Predicted_Sales_60': int(sum_60) if time_interval >= 60 else np.nan
        }
        # drop the rows with NaN values
        summary = {k: v for k, v in summary.items() if not pd.isna(v)}
        # Append the summary
        summary_list.append(summary)
    summarized_predictions = pd.DataFrame(summary_list)
    return summarized_predictions
