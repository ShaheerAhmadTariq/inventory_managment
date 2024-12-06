from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from io import StringIO
import pandas as pd

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
async def predict(category: str, time_interval: int):
    try:
        # Read the CSV data into a pandas DataFrame
        df = pd.read_csv('./sku_df_with_predictions.csv')

        # Convert the filtered DataFrame to a CSV string
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Return the CSV data as a response
        return StreamingResponse(csv_buffer, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=filtered_data.csv"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
