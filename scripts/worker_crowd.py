import asyncio, io, json, subprocess
from datetime import datetime
from google.cloud import vision, bigquery, visionai_v1 as va
from fastapi import FastAPI
import socketio

# --- CONFIG ---
PROJECT = "regal-river-463309-a3"
LOCATION = "us-central1"
STREAM_ID = "gate1-stream"
BQ_DATASET = "crowd_counts"
BQ_TABLE   = "live_counts"

# --- Initialize clients once ---
vision_client = vision.ImageAnnotatorClient()
bq_client     = bigquery.Client()
sio           = socketio.AsyncServer(async_mode="asgi")
app           = FastAPI()
sio_app       = socketio.ASGIApp(sio, app)

# BigQuery table reference
table_ref = bq_client.dataset(BQ_DATASET).table(BQ_TABLE)

# Ensure table exists with schema: timestamp:TIMESTAMP, count:INTEGER
# (Omitted: creation logic)

# --- WebSocket Endpoint ---
@app.get("/")
async def index():
    return {"message": "Crowd count service running"}

@sio.event
async def connect(sid, environ):
    print("Client connected:", sid)

@sio.event
async def disconnect(sid):
    print("Client disconnected:", sid)

# --- Stream Consumer & Processor ---
async def consume_and_process():
    # Run vaictl CLI to receive raw packets
    cmd = [
      "vaictl", "-p", PROJECT, "-l", LOCATION,
      "-c", "application-cluster-0",
      "receive", "streams", "packets", STREAM_ID
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE
    )
    frame_counter = 0
    async for line in proc.stdout:
        data = json.loads(line.decode())
        image_bytes = data.get("frame", {}).get("imageBytes")
        if not image_bytes:
            continue
        frame_counter += 1
        if frame_counter % 10 != 0:
            continue

        # 1. Convert to Vision API image
        image = vision.Image(content=image_bytes)
        # 2. Detect objects
        response = vision_client.object_localization(image=image)
        count = sum(1 for obj in response.localized_object_annotations
                    if obj.name.lower() == "person")

        timestamp = datetime.utcnow().isoformat()
        # 3. Stream insert to BigQuery
        row = {"insertId": data["frame"]["frameId"], 
               "json": {"timestamp": timestamp, "count": count}}
        errors = bq_client.insert_rows_json(table_ref, [row])
        if errors:
            print("BQ Insert errors:", errors)

        # 4. Emit via WebSocket
        await sio.emit("count_update", {"timestamp": timestamp, "count": count})

# --- Main Entrypoint ---
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(consume_and_process())
    import uvicorn
    uvicorn.run(sio_app, host="0.0.0.0", port=8000)
