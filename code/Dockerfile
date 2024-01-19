FROM python:3.9.18-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy
COPY ["predict.py", "embeddings.parquet", "./"]

COPY ml-zoomcamp-qa-sentence-transformer /app/ml-zoomcamp-qa-sentence-transformer

EXPOSE 9696

#ENTRYPOINT ["/bin/bash"]
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
