# pip install google-cloud google-cloud-vision google-cloud-pubsub google-cloud-bigquery
# Uses the VM's default service account credentials (no explicit key file needed).
# If running outside a GCP VM, set GOOGLE_APPLICATION_CREDENTIALS to a key file.

from google.cloud import bigquery
import sys

project_id = "ml-model-runner-f22694eb"
table_id = "ml-model-runner-f22694eb.lens_dgx_spark_dev.results"

# Optionally filter by run_id prefix: python query_bq.py <run_id>
run_id_prefix = sys.argv[1] if len(sys.argv) > 1 else None

client = bigquery.Client(project=project_id)

if run_id_prefix:
    query = f"""
        SELECT *
        FROM `{table_id}`
        WHERE STARTS_WITH(run_id, @prefix)
        ORDER BY processed_at DESC
        LIMIT 50
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("prefix", "STRING", run_id_prefix)]
    )
    print(f"Querying rows with run_id starting with: {run_id_prefix}")
else:
    query = f"""
        SELECT *
        FROM `{table_id}`
        ORDER BY processed_at DESC
        LIMIT 10
    """
    job_config = bigquery.QueryJobConfig()
    print("Querying 10 most recent rows...")

rows = list(client.query(query, job_config=job_config).result())
print(f"Got {len(rows)} rows\n")
for r in rows:
    print(dict(r))
