# pip install google-cloud google-cloud-vision google-cloud-pubsub google-cloud-bigquery
# The DGX Spark SA does not have permissions to insert values.
# Run "gcloud auth login" before running this to use your account perms :)

import json
import uuid

from google.cloud import pubsub_v1

project_id = "ml-model-runner-f22694eb"
topic_id = "lens-dgx-spark-dev-input"

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

run_id = uuid.uuid4().hex

print(run_id)

body_to_send = {
    "run_id": run_id,
    "letter_body": "Some letter text body...",
}
body_str = json.dumps(body_to_send).encode("utf-8")

# Store the future with image_blob_gs_path as key
future = publisher.publish(topic_path, body_str)
print(future.result(timeout=5))
