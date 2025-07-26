from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
# Assume face_matcher is your implementation
#from ..your_face_module import find_last_seen as face_find
from ..utils.your_face_moudle import find_last_seen as face_find
lostfound_tool = FunctionTool(
    face_find
)

lostfound_agent = LlmAgent(
    name="lost_and_found",
    model="gemini-2.5-pro",
    description="Locate last seen occurrence of a face",
    instruction=(
        "Given image_url, call find_last_seen(image_url) "
        "and return its JSON output."
    ),
    tools=[lostfound_tool],
)

# lostfound_runner = Runner(
#     agent=lostfound_agent, app_name="drishti_lostfound", session_service=session_service
# )

