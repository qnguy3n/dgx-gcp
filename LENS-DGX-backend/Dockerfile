FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicit file list rather than `COPY . .` so nothing in .dockerignore can
# accidentally sneak in regardless (credentials/, .venv/, logs/, etc.).
COPY config.py consume.py create_table.py csv_parse.py infer.py main.py test_logger.py write_bq.py ./
COPY tools/ tools/

# Unbuffered stdout so `docker logs` shows lines as they're written, not
# only on buffer flush/exit.
ENV PYTHONUNBUFFERED=1
ENV LENS_DGX_ENV=dev

# No secrets baked in: credentials/ and logs/ are bind-mounted at `docker run`
# time (see README's Docker section).
CMD ["python3", "main.py"]
