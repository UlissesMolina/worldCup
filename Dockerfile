FROM public.ecr.aws/lambda/python:3.11 AS builder

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -t /opt/python

FROM public.ecr.aws/lambda/python:3.11

COPY --from=builder /opt/python ${LAMBDA_TASK_ROOT}

# Copy application code
COPY src/ ${LAMBDA_TASK_ROOT}/src/

# Copy model artifact and processed data
COPY models/model.pkl ${LAMBDA_TASK_ROOT}/models/model.pkl
COPY models/metadata.txt ${LAMBDA_TASK_ROOT}/models/metadata.txt
COPY data/processed/matches.parquet ${LAMBDA_TASK_ROOT}/data/processed/matches.parquet

CMD ["src.api.lambda_handler.handler"]
