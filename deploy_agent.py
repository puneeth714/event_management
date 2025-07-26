import vertexai
from dristi_adk_agent.agent import root_agent
from vertexai.preview import reasoning_engines

PROJECT_ID = "regal-river-463309-a3"
LOCATION = "us-central1"
STAGING_BUCKET = "gs://agent-bucket-event-manage"

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=STAGING_BUCKET,
)

app = reasoning_engines.AdkApp(
    agent=root_agent,
    enable_tracing=True,
)
session = app.create_session(user_id="u_123")

session = app.get_session(user_id="u_123", session_id=session.id)

for event in app.stream_query(
    user_id="u_123",
    session_id=session.id,
    message="what you can do ?",
):
    print(event)

from vertexai import agent_engines

remote_app = agent_engines.create(
    agent_engine=app,
    requirements=[
        "google-cloud-aiplatform[adk,agent_engines]"   
    ]
)

remote_session = remote_app.create_session(user_id="u_456")
for event in remote_app.stream_query(
    user_id="u_456",
    session_id=remote_session["id"],
    message="whats the weather in new york",
):
    print(event)