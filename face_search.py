"""
End-to-end face search in a video using InspireFace.

1.  Extract a 512-d reference embedding from probe.jpg.
2.  Iterate through every frame in target.mp4.
3.  For each frame:
        a. Detect faces
        b. Align & extract embeddings
        c. Compare cosine-similarity to ALL reference embeddings
4.  Log every matching time-span above the threshold.
5.  (Optional) Draw bounding boxes and write annotated video.

Author: Your Name – 2025-07-24
"""

from __future__ import annotations
import cv2
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import inspireface as isf  # pip install inspireface
# ------------------------- CONFIGURATION ------------------------- #
# Paths
ROOT = Path(__file__).resolve().parent
PHOTO_PATH = ROOT / "input" / "face_to_detect_2.png"
VIDEO_PATH = ROOT / "input" / "target.mp4"
OUT_CSV = ROOT / "output" / "hits.csv"
ANNOTATED_VIDEO = None

# Thresholds
COSINE_THRESHOLD = 0.35      # lower → stricter. 0.35 ≈ 99.7% TAR on LFW[50]
MIN_FRAMES_PER_HIT = 3      # consecutive frames to confirm a match
MAX_MISSES_ALLOWED = 5      # drop track after n misses

# Runtime
DETECT_EVERY_N_FRAMES = 1   # set >1 to speed up at the cost of recall
MAX_DISPLAY_FPS = 30        # cap for annotated video
# ------------------------------------------------------------------ #


def init_session() -> isf.InspireFaceSession:
    """
    Launch inspireface runtime and create a session with detection+recognition.
    """
    if not isf.launch():
        print("Failed to initialize InspireFace resources.", file=sys.stderr)
        sys.exit(1)

    # enable face detection + feature extraction
    flags = isf.HF_DETECT_MODE_ALWAYS_DETECT | isf.HF_ENABLE_FACE_RECOGNITION
    session = isf.InspireFaceSession(flags, isf.HF_DETECT_MODE_ALWAYS_DETECT)
    return session


def get_reference_embeddings(session: isf.InspireFaceSession) -> List[np.ndarray]:
    """
    Detect all faces in PHOTO_PATH and return a list of their 512-d embeddings.
    """
    img = cv2.imread(str(PHOTO_PATH))
    if img is None:
        raise FileNotFoundError(PHOTO_PATH)

    faces = session.face_detection(img)
    if not faces:
        raise RuntimeError("No face found in reference photo.")

    reference_embeddings = []
    for fobj in faces:
        embedding = session.face_feature_extract(image=img, face_information=fobj)
        if embedding is not None: # Ensure embedding was successfully extracted
            reference_embeddings.append(embedding / np.linalg.norm(embedding))  # L2-normalise
    
    if not reference_embeddings:
        raise RuntimeError("Could not extract any embeddings from the reference photo.")

    return reference_embeddings


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return 1-cosine_similarity (so 0 = identical)."""
    # Ensure inputs are normalized before calculating distance
    # np.dot handles 1D arrays correctly.
    return 1 - float(np.dot(a, b))


class Tracker:
    """
    Simple state machine to accumulate consecutive hits into time-spans.
    """
    def __init__(self, fps: float):
        self.fps = fps
        self.current_hit: List[int] = []   # frames in the ongoing hit
        self.all_hits: List[Tuple[int, int]] = []  # (start_frame, end_frame)
        self.misses = 0

    def update(self, frame_idx: int, is_hit: bool):
        if is_hit:
            self.current_hit.append(frame_idx)
            self.misses = 0
        else:
            self.misses += 1

        # If we've missed too many frames, conclude the current hit
        if self.current_hit and (not is_hit and self.misses >= MAX_MISSES_ALLOWED):
            self.finalize_hit()

    def finalize_hit(self):
        if len(self.current_hit) >= MIN_FRAMES_PER_HIT:
            self.all_hits.append((self.current_hit[0], self.current_hit[-1]))
        self.current_hit.clear()
        self.misses = 0

    def close(self):
        self.finalize_hit()

    def to_csv(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["start_sec", "end_sec", "duration_sec",
                             "start_frame", "end_frame"])
            for start, end in self.all_hits:
                start_t = start / self.fps
                end_t = end / self.fps
                writer.writerow([f"{start_t:.3f}", f"{end_t:.3f}",
                                 f"{end_t-start_t:.3f}", start, end])


def main():
    session = init_session()
    ref_embeddings = get_reference_embeddings(session) # Now a list of embeddings
    print(f"Extracted {len(ref_embeddings)} reference embedding(s).")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare video writer if annotation requested
    writer = None
    if ANNOTATED_VIDEO:
        ANNOTATED_VIDEO.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(ANNOTATED_VIDEO), fourcc,
                                 min(fps, MAX_DISPLAY_FPS), (width, height))

    tracker = Tracker(fps=fps)

    pbar = tqdm(total=total_frames, desc="Scanning video")
    frame_idx = -1
    last_detection_boxes: List[Tuple[int, int, int, int]] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        pbar.update(1)

        is_match_in_frame = False
        
        # Only perform detection if DETECT_EVERY_N_FRAMES allows
        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            faces_in_frame = session.face_detection(frame)
            last_detection_boxes = [] # Reset boxes for this frame

            for fobj in faces_in_frame:
                emb = session.face_feature_extract(image=frame, face_information=fobj)
                
                if emb is not None:
                    emb = emb / np.linalg.norm(emb) # L2-normalise
                    
                    # Compare against ALL reference embeddings
                    for ref_emb in ref_embeddings:
                        dist = cosine_distance(ref_emb, emb)

                        if dist < COSINE_THRESHOLD:
                            is_match_in_frame = True
                            # Store box for drawing if it's a match
                            x1, y1, x2, y2 = fobj.location
                            last_detection_boxes.append((x1, y1, x2, y2))
                            # No need to break here, we want to find all matching faces in this frame
                            break # Break from inner loop (comparing to ref_embeddings) once a match is found for this face
                
        # Update tracker based on whether ANY face in this frame matched ANY reference face
        tracker.update(frame_idx, is_match_in_frame)

        # Draw & write frame if needed
        if writer is not None:
            vis = frame.copy()
            for (x1, y1, x2, y2) in last_detection_boxes:
                # Always draw green for matched faces
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, "MATCH", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # If no matches in this frame but there were detections in previous frames (due to DETECT_EVERY_N_FRAMES > 1)
            # you might want to consider drawing previous boxes or changing the logic.
            # For simplicity, if DETECT_EVERY_N_FRAMES > 1, boxes will only appear on detection frames.
            
            writer.write(vis)

    # Flush
    tracker.close()
    cap.release()
    if writer:
        writer.release()
    pbar.close()

    # Export results
    tracker.to_csv(OUT_CSV)
    print(f"\nFinished. {len(tracker.all_hits)} hit(s) saved to {OUT_CSV}")
    if ANNOTATED_VIDEO:
        print(f"Annotated video: {ANNOTATED_VIDEO}")


if __name__ == "__main__":
    main()