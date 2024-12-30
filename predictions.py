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
    sku_columns = ['Inventory_Item_Internal_ID', 'Inventory_Location_Abbreviation']
    unique_skus = filtered_df[sku_columns].drop_duplicates()

    # Initialize a list to store summary dictionaries for each SKU
    summary_list = []

    # Iterate over each unique SKU
    for idx, sku in unique_skus.iterrows():
        # Create a filter for the current SKU
        sku_filter = (
            (filtered_df['Inventory_Item_Internal_ID'] == sku['Inventory_Item_Internal_ID']) &
            (filtered_df['Inventory_Location_Abbreviation'] == sku['Inventory_Location_Abbreviation'])
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

        predictions = np.where(predictions <= 1.5, 0, np.round(predictions))
        # Sum the predictions for each week (1st, 2nd, ..., 8th)
        sum_1 = predictions[:7].sum()  # Week 1 (1st week = 7 days)
        sum_2 = predictions[7:14].sum()  # Week 2 (2nd week = days 8-14)
        sum_3 = predictions[14:21].sum()  # Week 3 (3rd week = days 15-21)
        sum_4 = predictions[21:28].sum()  # Week 4 (4th week = days 22-28)
        sum_5 = predictions[28:35].sum()  # Week 5 (5th week = days 29-35)
        sum_6 = predictions[35:42].sum()  # Week 6 (6th week = days 36-42)
        sum_7 = predictions[42:49].sum()  # Week 7 (7th week = days 43-49)
        sum_8 = predictions[49:56].sum()  # Week 8 (8th week = days 50-56)

        # Create a summary
        summary = {
            'Inventory_Item_Internal_ID': sku['Inventory_Item_Internal_ID'],
            'Inventory_Location_Abbreviation': sku['Inventory_Location_Abbreviation'],
            'Inventory_Size': X_new['Inventory_Size'].iloc[0],
            'Inventory_Themes': X_new['Inventory_Themes'].iloc[0],
            'Inventory_MH3_Class': X_new['Inventory_MH3_Class'].iloc[0],
            'Inventory_MH1_Division': X_new['Inventory_MH1_Division'].iloc[0],
            'Inventory_Lifestyle_Category': X_new['Inventory_Lifestyle_Category'].iloc[0],
            'Inventory_Product_Category': product_category,
            'Predicted_Sales_Week_1': int(sum_1),
            'Predicted_Sales_Week_2': int(sum_2),
            'Predicted_Sales_Week_3': int(sum_3),
            'Predicted_Sales_Week_4': int(sum_4),
            'Predicted_Sales_Week_5': int(sum_5),
            'Predicted_Sales_Week_6': int(sum_6),
            'Predicted_Sales_Week_7': int(sum_7),
            'Predicted_Sales_Week_8': int(sum_8)
        }

        # Append the summary
        summary_list.append(summary)

        # Convert the list to a DataFrame
    summarized_predictions = pd.DataFrame(summary_list)
    return summarized_predictions

