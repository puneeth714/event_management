from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.cloud import bigquery
from google.cloud import aiplatform

# Tool: Detect anomalies using forecasting model
def detect_anomaly(start_time: str, end_time: str) -> dict:
    # 1. Fetch actual counts
    client = bigquery.Client()
    q = """
      SELECT ingestion_time, COUNT(*) AS count
      FROM `YOUR_PROJECT_ID.drishti_ds.crowd_events_geo`
      WHERE ingestion_time BETWEEN @start AND @end
      GROUP BY ingestion_time
      ORDER BY ingestion_time
    """
    job = client.query(
        q,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start", "TIMESTAMP", start_time),
                bigquery.ScalarQueryParameter("end", "TIMESTAMP", end_time),
            ]
        ),
    )
    actual = [dict(r) for r in job]

    # 2. Predict baseline
    endpoint = aiplatform.Endpoint("projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/ANOMALY_MODEL_ID")
    predictions = endpoint.predict(instances=[{"timestamp": r["ingestion_time"].isoformat()} for r in actual])
    baseline = [p.predictions[0] for p in predictions.predictions]

    # 3. Compare
    anomalies = []
    for a, b in zip(actual, baseline):
        if a["count"] > b * 1.5:
            anomalies.append({"time": a["ingestion_time"].isoformat(), "actual": a["count"], "expected": b})

    return {"anomalies": anomalies}

anomaly_tool = FunctionTool(
    detect_anomaly
)

anomaly_agent = LlmAgent(
    name="anomaly_detector",
    model="gemini-2.5-pro",
    description="Flag spikes in crowd density",
    instruction=(
        "Given start_time and end_time, call detect_anomaly(start_time, end_time) "
        "and return the anomalies JSON."
    ),
    tools=[anomaly_tool],
)

# anomaly_runner = Runner(
#     agent=anomaly_agent, app_name="drishti_anomaly", session_service=session_service
# )

# timeseries forecasting - for anomaly detection.