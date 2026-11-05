# pip install google-cloud google-cloud-vision google-cloud-pubsub google-cloud-bigquery
# The dev_auth.json file contains the credentials this machine should use

from google.cloud import bigquery
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<path>/dev_auth.json"

# Construct a BigQuery client object.
client = bigquery.Client()

table_id = "ml-model-runner-f22694eb.lens_dgx_spark_dev.codcot_example"

schema = [
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("run_length", "INTEGER", mode="REQUIRED"),
]

table = bigquery.Table(table_id, schema=schema)
table = client.create_table(table, exists_ok=True)  # Make an API request.
print(
    "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
)

client.insert_rows(table, [
    ("runid1", "123"),
    ("runid2", "234"),
])

print ("Inserted rows")