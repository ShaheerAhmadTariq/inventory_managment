# FastAPI ML Model Inference API

This is a FastAPI application that serves a machine learning model for sales prediction based on inventory and transaction data. It exposes a REST API for making predictions on input data in JSON format.

## Prerequisites

- Python 3.11.5 (or higher)
- Docker (for containerized deployment)

## Setup Instructions

### 1. Clone the Repository

Clone this repository to your local machine.

```bash
git clone https://github.com/yourusername/fastapi-ml-model-api.git
cd fastapi-ml-model-api
```

## For Linux/macOS
```
python3 -m venv venv
source venv/bin/activate
```

## For Windows
```
python -m venv venv
venv\Scripts\activate
```

## Install the required Python dependencies using pip
```
pip install -r requirements.txt
```

## Start the FastAPI server using uvicorn. This will run the server locally.
```
uvicorn app:app --reload
```

