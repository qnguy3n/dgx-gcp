# pip install google-cloud google-cloud-vision google-cloud-pubsub google-cloud-bigquery
# The dev_auth.json file contains the credentials this machine should use

from google.cloud import pubsub_v1
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<path>/dev_auth.json"

project_id = "ml-model-runner-f22694eb"
subscription_id = "lens-dgx-spark-dev-input"
timeout = None  # 5.0

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)


def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    print(f"Received {message}.")
    message.ack()


streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
print(f"Listening for messages on {subscription_path}..\n")

# Wrap subscriber in a 'with' block to automatically call close() when done.
with subscriber:
    try:
        # When `timeout` is not set, result() will block indefinitely,
        # unless an exception is encountered first.
        streaming_pull_future.result(timeout=timeout)
    except TimeoutError:
        streaming_pull_future.cancel()  # Trigger the shutdown.
        streaming_pull_future.result()  # Block until the shutdown is complete.
