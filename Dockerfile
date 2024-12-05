# Use official Python 3.11.5 image from Docker Hub
FROM python:3.11.5-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt /app/

# Create a virtual environment in the container
RUN python3 -m venv /app/venv

# Ensure pip is up to date
RUN /app/venv/bin/pip install --upgrade pip

# Install the required dependencies inside the virtual environment
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . /app/

# Set environment variables to use the virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Command to run the FastAPI app using uvicorn from the virtual environment
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
