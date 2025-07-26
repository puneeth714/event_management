#!/usr/bin/env python3
import os
from google.cloud import videointelligence
from google.cloud import bigquery
from datetime import timedelta

# Initialize clients
video_client = videointelligence.VideoIntelligenceServiceClient()
bq_client = bigquery.Client()

# Parameters
GCS_URI = "gs://agent-bucket-event-manage/send1.mp4"

BQ_OBJECT_TABLE = f"{os.environ['PROJECT_ID']}.video_ai.object_annotations"
BQ_FACE_TABLE   = f"{os.environ['PROJECT_ID']}.video_ai.face_annotations"

def load_annotation_results(uri):
    # Request object and face tracking
    features = [
        videointelligence.Feature.OBJECT_TRACKING,
        videointelligence.Feature.FACE_DETECTION
    ]
    request = {
      "input_uri": uri,
      "features": features,
      "video_context": {
        "object_tracking_config": {},
        "face_detection_config": {}
      }
    }
    operation = video_client.annotate_video(request=request)
    print(f"Annotation running for {uri}...")
    result = operation.result(timeout=600)
    return result.annotation_results[0]

def record_object_annotations(annotations, uri):
    rows = []
    for obj in annotations.object_annotations:
        start = obj.segment.start_time_offset
        end   = obj.segment.end_time_offset
        for frame in obj.frames[:1]:  # first frame box
            box = frame.normalized_bounding_box
            rows.append({
                "input_uri": uri,
                "segment_start": timedelta(seconds=start.seconds, microseconds=start.nanos/1000),
                "segment_end":   timedelta(seconds=end.seconds,   microseconds=end.nanos/1000),
                "entity":        obj.entity.description,
                "confidence":    obj.confidence,
                "track_id":      obj.track_id,
                "box_LEFT":      box.left,
                "box_TOP":       box.top,
                "box_RIGHT":     box.right,
                "box_BOTTOM":    box.bottom
            })
    if rows:
        bq_client.insert_rows_json(BQ_OBJECT_TABLE, rows)
        print(f"Wrote {len(rows)} object rows to BigQuery")

def record_face_annotations(annotations, uri):
    rows = []
    for face in annotations.face_detection_annotations:
        start = face.segment.start_time_offset
        end   = face.segment.end_time_offset
        box   = face.frames[0].normalized_bounding_box
        rows.append({
            "input_uri": uri,
            "segment_start": timedelta(seconds=start.seconds, microseconds=start.nanos/1000),
            "segment_end":   timedelta(seconds=end.seconds,   microseconds=end.nanos/1000),
            "confidence":    face.confidence,
            "box_LEFT":      box.left,
            "box_TOP":       box.top,
            "box_RIGHT":     box.right,
            "box_BOTTOM":    box.bottom
        })
    if rows:
        bq_client.insert_rows_json(BQ_FACE_TABLE, rows)
        print(f"Wrote {len(rows)} face rows to BigQuery")

def main():
    result = load_annotation_results(GCS_URI)
    record_object_annotations(result, GCS_URI)
    record_face_annotations(result, GCS_URI)
    print("Annotation complete.")

if __name__ == "__main__":
    main()
