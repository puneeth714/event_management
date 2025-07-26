from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from .sub_agents.crowd_agents import crowd_analysis_agent#, crowd_runner
from .sub_agents.anamoly_agents import anomaly_agent#, anomaly_runner
from .sub_agents.lostfound_agents import lostfound_agent#, lostfound_runner
from .sub_agents.preevent_agents import preevent_agent#, preevent_runner

#load env
import os
from dotenv import load_dotenv
load_dotenv()

root_agent = LlmAgent(
    name="central_manager",
    model="gemini-2.5-flash",
    description="Orchestrates all Drishti agents.",
    instruction=(
        "Route tasks to crowd_analysis, anomaly_detector, lost_and_found, or pre_event_planner "
        "based on user intent. Return only the tool output JSON."
    ),
    sub_agents=[crowd_analysis_agent, anomaly_agent, lostfound_agent, preevent_agent],
)

# central_runner = Runner(
#     agent=root_agent, app_name="drishti_central", session_service=InMemorySessionService()
# )
