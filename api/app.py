from flask import Flask, request, jsonify, send_file
from flask_restx import Api, Resource, fields
from io import StringIO
import pandas as pd
from data_models import InventoryData, SalesPrediction

app = Flask(__name__)

# Create an API object
api = Api(app, doc='/swagger')  # Swagger UI will be available at /swagger

# Define a model for the request payload using Flask-RESTX
inventory_data_model = api.model('InventoryData', {
    'sku': fields.String(required=True, description='The SKU of the product'),
    'category': fields.String(required=True, description='The product category'),
    'quantity': fields.Integer(required=True, description='The quantity available')
})

# Define a model for the prediction response
sales_prediction_model = api.model('SalesPrediction', {
    'sales_prediction': fields.Float(required=True, description='The predicted sales')
})


@app.route('/')
def home():
    return "Inventory Managment API"
# Inference endpoint
@api.route('/single-sku-predict/')
class PredictSingleSKU(Resource):
    @api.expect(inventory_data_model)  # Expect data in the format of InventoryData
    @api.marshal_with(sales_prediction_model)  # Marshal output to SalesPrediction
    def post(self):
        try:
            # Parse the incoming data
            data = request.get_json()
            inventory_data = InventoryData(**data)

            # Create a prediction (just a placeholder here)
            prediction = SalesPrediction(sales_prediction=1.0)

            return prediction
        except Exception as e:
            return {'error': str(e)}, 500


# CSV file endpoint
@api.route('/predict')
@api.param('category', 'Category to filter the SKU data by')
class PredictCSV(Resource):
    def get(self):
        # Get the category parameter from the query string
        category = request.args.get('category', None)

        if category is None:
            return {'error': 'Category parameter is required'}, 400

        try:
            # Return the filtered CSV data as a file response
            return send_file('./sku_df_with_predictions.csv', as_attachment=True, download_name="sales_prediction.csv", mimetype="text/csv")

        except Exception as e:
            return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)