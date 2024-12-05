from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class InventoryData(BaseModel):
    inventory_item_internal_id: int
    inventory_item_name_number: str
    inventory_product_category: Optional[str] = None
    inventory_location_abbreviation: str
    inventory_capsules: Optional[str] = None
    inventory_themes: Optional[str] = None
    inventory_mh3_class: Optional[str] = None
    inventory_mh1_division: Optional[str] = None
    inventory_lifestyle_category: Optional[str] = None
    inventory_size: Optional[str] = None
    inventory_base_price: float
    inventory_original_price: Optional[float] = None
    transactions_transaction_number: str
    date_transaction_date: datetime
    transactions_net_sales_for_product: float

class SalesPrediction(BaseModel):
    sales_prediction: float
