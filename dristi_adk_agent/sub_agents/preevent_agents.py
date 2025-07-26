from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import googlemaps
import random
import os
# load environment variables
from dotenv import load_dotenv
load_dotenv()
gmaps = googlemaps.Client(key=os.getenv("NEXT_PUBLIC_GOOGLE_MAPS_API_KEY"))

def plan_pre_event(map_image_url: str, cameras: list, expected_attendance: int) -> dict:
    # 1. Georeference (stub)
    # 2. Simulate coverage: random example
    placements = []
    for cam in cameras:
        placements.append({"camera": cam["id"], "lat": cam["lat"], "lng": cam["lng"]})
    # 3. Simulate staff routes (stub)
    staff_plan = [{"staff_id": i, "route": []} for i in range(1, 6)]
    return {"placements": placements, "staff_plan": staff_plan}

preevent_tool = FunctionTool(
    plan_pre_event
)

preevent_agent = LlmAgent(
    name="pre_event_planner",
    model="gemini-2.5-pro",
    description="Plan camera and staff positions given map and expected guests",
    instruction=(
        "Given map_image_url, cameras list, and expected_attendance, "
        "call plan_pre_event(...) and return the plan JSON."
    ),
    tools=[preevent_tool],
)

# preevent_runner = Runner(
#     agent=preevent_agent, app_name="drishti_preevent", session_service=session_service
# )
