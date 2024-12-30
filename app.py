from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from io import StringIO
import pandas as pd
from utils import read_data, get_valid_categories
from predictions import inference
from mangum import Mangum  # Import Mangum

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/', tags=["Root"])
async def health_check():
    return {"message":"Ineventory Management API is up and running"}


@app.get("/predict")
async def predict(category: str, time_interval: int | None = None):
    try:
        # check if the category is valid

        if category not in get_valid_categories():
            raise ValueError(f"Invalid category. Please choose from: {get_valid_categories()}")

        # if time_interval not in params then it is 60
        time_interval = time_interval if time_interval else 60

        df = read_data('./processed_netsuite_sales_data.csv')

        predictions_df = inference(df, category, time_interval)
        # Convert the filtered DataFrame to a CSV string
        csv_buffer = StringIO()
        predictions_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Return the CSV data as a response
        return StreamingResponse(csv_buffer, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

handler = Mangum(app)  # Create a handler for AWS Lambda