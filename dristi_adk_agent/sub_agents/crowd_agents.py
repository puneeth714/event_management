from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.cloud import bigquery
import json
#from camera_locations import CAMERA_LOCATIONS
from ..utils.camera_locations import CAMERA_LOCATIONS
# Tool: Query BigQuery for crowd events
def query_crowd(start_time: str, end_time: str, bounds_geojson: str) -> dict:
    client = bigquery.Client()
    sql = """
      SELECT ST_X(geo) AS lng, ST_Y(geo) AS lat, COUNT(*) AS count
      FROM `YOUR_PROJECT_ID.drishti_ds.crowd_events_geo`
      WHERE ingestion_time BETWEEN @start AND @end
        AND ST_WITHIN(geo, ST_GEOGFROMGEOJSON(@bounds))
      GROUP BY lng, lat
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start", "TIMESTAMP", start_time),
                bigquery.ScalarQueryParameter("end", "TIMESTAMP", end_time),
                bigquery.ScalarQueryParameter("bounds", "STRING", bounds_geojson),
            ]
        ),
    )
    rows = [dict(row) for row in job]
    return {"points": rows}

crowd_tool = FunctionTool(
    query_crowd
)

crowd_analysis_agent = LlmAgent(
    name="crowd_analysis",
    model="gemini-2.5-pro",  
    description="Queries crowd density from BigQuery",
    instruction=(
        "Given start_time, end_time, and bounds_geojson, "
        "call query_crowd(start_time, end_time, bounds_geojson) "
        "and return its JSON output."
    ),
    tools=[crowd_tool],
)

# session_service = InMemorySessionService()
# crowd_runner = Runner(
#     agent=crowd_analysis_agent, app_name="drishti_crowd", session_service=session_service
# )
