from google.cloud import videointelligence
import argparse

def detect_faces(video_uri):
    """Detect faces in a video using Google Cloud Video Intelligence API."""
    client = videointelligence.VideoIntelligenceServiceClient()

    operation = client.annotate_video(
        request={
            "features": [videointelligence.Feature.FACE_DETECTION],
            "input_uri": video_uri,
        }
    )

    print("\nProcessing video for face detection...")
    result = operation.result(timeout=90)
    print("Processing completed.\n")

    # Process the annotations
    annotations = result.annotation_results[0]
    for annotation in annotations.face_detection_annotations:
        for track in annotation.tracks:
            print(f"Face detected from {track.segment.start_time_offset.seconds}s")
            print(f"    to {track.segment.end_time_offset.seconds}s")
            print(f"    Confidence: {track.confidence:.2%}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_uri", help="GCS URI or local path to video file (e.g. gs://bucket/file.mp4 or file.mp4)")
    args = parser.parse_args()
    
    detect_faces(args.video_uri)
