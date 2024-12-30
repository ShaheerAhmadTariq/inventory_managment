# Use the official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.11

# Set the working directory in the container
# WORKDIR /app
WORKDIR /var/task

# Copy the application requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy the rest of the application code
COPY . .

# Set the correct entrypoint for the Lambda runtime
# ENTRYPOINT ["/lambda-entrypoint.sh", "app.handler"]
CMD ["app.handler"]

