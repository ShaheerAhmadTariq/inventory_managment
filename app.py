from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from data_models import InventoryData, SalesPrediction
from io import StringIO
import pandas as pd

app = FastAPI()


# Inference endpoint
@app.post("/single-sku-predict/", response_model=SalesPrediction)
async def predict_single_sku(data: InventoryData):
    input_data = pd.DataFrame([data.dict()])
    try:
        return SalesPrediction(sales_prediction=1.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict")
async def predict(category: str):
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
