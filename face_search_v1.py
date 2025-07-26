"""
Improved multi-face search in crowded video scenes using InspireFace.
Fixes early detection issues for small/distant faces and annotates ALL detected faces.

This script:
1. Extracts a reference embedding from a probe image.
2. Processes each video frame with enhanced preprocessing for better small-face detection.
3. Detects and recognizes all faces, tracking matches.
4. Annotates EVERY detected face in the output video (matches in green, others in blue).
5. Outputs a CSV with tracked time-spans for matches.

Author: AI Assistant – 2025-07-24
"""

from __future__ import annotations
import cv2
import csv
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

import inspireface as isf  # pip install inspireface

# ------------------------- CONFIGURATION ------------------------- #
# Paths
ROOT = Path(__file__).resolve().parent
PHOTO_PATH = ROOT / "input" / "probe.jpg"
VIDEO_PATH = ROOT / "input" / "target.mp4"
OUT_CSV = ROOT / "output" / "hits.csv"
ANNOTATED_VIDEO = ROOT / "output" / "annotated.mp4"  # set to None to skip

# Thresholds and Parameters
COSINE_THRESHOLD = 0.4  # For matching
MIN_FRAMES_PER_HIT = 5  # For tracking confirmation
MAX_MISSES_ALLOWED = 8  # For handling occlusions
DETECT_EVERY_N_FRAMES = 1  # Set >1 for speed
MAX_DISPLAY_FPS = 30  # For annotated video
ENABLE_PREPROCESSING = True
ENABLE_MULTI_SCALE = True
MIN_FACE_SIZE = 2  # Lowered for smaller/distant faces
MAX_FACES_PER_FRAME = 500  # Limit to prevent overload
MULTI_SCALES = [0.5, 0.8, 1.0, 1.2, 1.5]  # Expanded for better small-face detection
# ------------------------------------------------------------------ #


def init_session() -> isf.InspireFaceSession:
    """
    Launch InspireFace runtime with video mode for better sensitivity.
    """
    if not isf.launch():
        print("Failed to initialize InspireFace resources.", file=sys.stderr)
        sys.exit(1)

    # Enable detection + recognition, use video mode for continuous tracking
    flags = isf.HF_DETECT_MODE_ALWAYS_DETECT | isf.HF_ENABLE_FACE_RECOGNITION
    session = isf.InspireFaceSession(flags)  # Changed to video mode
    return session


def get_reference_embedding(session: isf.InspireFaceSession) -> np.ndarray:
    """
    Detect the largest face in PHOTO_PATH and return its 512-d embedding.
    """
    img = cv2.imread(str(PHOTO_PATH))
    if img is None:
        raise FileNotFoundError(PHOTO_PATH)

    faces = session.face_detection(img)
    if not faces:
        raise RuntimeError("No face found in reference photo.")

    # Pick the face with largest area
    faces = sorted(faces, key=lambda f: (f.location[2] - f.location[0]) *
                   (f.location[3] - f.location[1]), reverse=True)
    embedding = session.face_feature_extract(img, faces[0])
    return embedding / np.linalg.norm(embedding)  # L2-normalise


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return 1 - cosine_similarity (so 0 = identical)."""
    return 1 - float(np.dot(a, b))


def enhance_frame_for_crowded_detection(frame):
    """Apply preprocessing optimized for crowded scenes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(enhanced.astype(np.float32), -1, kernel)
    local_std = cv2.filter2D((enhanced.astype(np.float32) - local_mean) ** 2, -1, kernel)
    local_std = np.sqrt(local_std + 1e-10)
    normalized = (enhanced.astype(np.float32) - local_mean) / local_std
    normalized = np.clip(normalized * 50 + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def multi_scale_preprocessing(frame, scales=MULTI_SCALES):
    """Process frame at multiple scales for better detection of small faces."""
    processed_frames = []
    for scale in scales:
        if scale != 1.0:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_frame = cv2.resize(frame, (new_w, new_h))
        else:
            scaled_frame = frame.copy()
        enhanced = enhance_frame_for_crowded_detection(scaled_frame)
        processed_frames.append((enhanced, scale))
    return processed_frames


def apply_illumination_normalization(frame):
    """Apply illumination-invariant preprocessing."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    def single_scale_retinex(img, sigma):
        return np.log10(img + 1.0) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1.0)
    scales = [15, 80, 250]
    weights = [1/3.0] * 3
    l_float = l.astype(np.float64) + 1.0
    retinex = np.zeros_like(l_float)
    for scale, weight in zip(scales, weights):
        retinex += weight * single_scale_retinex(l_float, scale)
    retinex = np.clip((retinex - retinex.min()) / (retinex.max() - retinex.min()) * 255, 0, 255)
    l_enhanced = retinex.astype(np.uint8)
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def remove_duplicate_detections(matches: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Remove duplicate face detections using IoU."""
    if not matches:
        return matches
    matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)
    filtered = []
    for current in matches:
        is_duplicate = False
        for existing in filtered:
            if calculate_iou(current['bbox'], existing['bbox']) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered.append(current)
    return filtered


def calculate_iou(bbox1: Tuple, bbox2: Tuple) -> float:
    """Calculate Intersection over Union."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)
    if x2_int <= x1_int or y2_int <= y1_int:
        return 0.0
    intersection = (x2_int - x1_int) * (y2_int - y1_int)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def enhanced_face_detection(session, frame, reference_embedding):
    """Detect and match all faces with improved sensitivity for small faces."""
    processed_frames = []
    if ENABLE_PREPROCESSING:
        frame = apply_illumination_normalization(frame)
        if ENABLE_MULTI_SCALE:
            processed_frames = multi_scale_preprocessing(frame)
        else:
            enhanced = enhance_frame_for_crowded_detection(frame)
            processed_frames = [(enhanced, 1.0)]
    else:
        processed_frames = [(frame, 1.0)]
    
    all_detections = []  # Store all detections, not just matches
    all_matches = []     # Store only matches for tracking
    
    for enhanced_frame, scale in processed_frames:
        try:
            faces = session.face_detection(enhanced_frame)
            for fobj in faces:
                try:
                    x1, y1, x2, y2 = fobj.location
                    if scale != 1.0:
                        x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)
                    face_size = (x2 - x1) * (y2 - y1)
                    if face_size < MIN_FACE_SIZE * MIN_FACE_SIZE:
                        continue
                    
                    # Extract embedding for matching
                    emb = session.face_feature_extract(enhanced_frame, fobj)
                    emb = emb / np.linalg.norm(emb)
                    dist = cosine_distance(reference_embedding, emb)
                    confidence = 1.0 - dist
                    
                    # Store all detections
                    all_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'is_match': dist < COSINE_THRESHOLD
                    })
                    
                    # Store matches for tracking
                    if dist < COSINE_THRESHOLD:
                        all_matches.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'face_size': face_size,
                            'scale': scale
                        })
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
        except Exception as e:
            print(f"Error in face detection: {e}")
            continue
    
    # Remove duplicates
    filtered_matches = remove_duplicate_detections(all_matches)
    filtered_detections = remove_duplicate_detections(all_detections)  # For annotations
    
    return filtered_matches, filtered_detections


class EnhancedMultiFaceTracker:
    def __init__(self, fps: float, max_faces_per_frame: int = MAX_FACES_PER_FRAME):
        self.fps = fps
        self.max_faces_per_frame = max_faces_per_frame
        self.all_tracks = []
        self.active_tracks = {}
        self.track_id_counter = 0
        
    def update(self, frame_idx: int, detected_faces: List[Dict]):
        matched_tracks = {}
        new_faces = []
        for face_data in detected_faces:
            best_match_id = None
            best_match_score = float('inf')
            for track_id, track_data in self.active_tracks.items():
                if frame_idx - track_data['last_frame'] > MAX_MISSES_ALLOWED:
                    continue
                bbox_dist = self._bbox_distance(face_data['bbox'], track_data['last_bbox'])
                if bbox_dist < best_match_score and bbox_dist < 100:
                    best_match_score = bbox_dist
                    best_match_id = track_id
            if best_match_id is not None:
                matched_tracks[best_match_id] = face_data
                self.active_tracks[best_match_id]['frames'].append(frame_idx)
                self.active_tracks[best_match_id]['last_frame'] = frame_idx
                self.active_tracks[best_match_id]['last_bbox'] = face_data['bbox']
                self.active_tracks[best_match_id]['confidences'].append(face_data['confidence'])
            else:
                new_faces.append(face_data)
        for face_data in new_faces:
            if len(self.active_tracks) < self.max_faces_per_frame:
                track_id = self.track_id_counter
                self.track_id_counter += 1
                self.active_tracks[track_id] = {
                    'frames': [frame_idx],
                    'start_frame': frame_idx,
                    'last_frame': frame_idx,
                    'last_bbox': face_data['bbox'],
                    'confidences': [face_data['confidence']]
                }
        expired_tracks = []
        for track_id, track_data in self.active_tracks.items():
            if frame_idx - track_data['last_frame'] > MAX_MISSES_ALLOWED:
                if len(track_data['frames']) >= MIN_FRAMES_PER_HIT:
                    self.all_tracks.append({
                        'start_frame': track_data['start_frame'],
                        'end_frame': track_data['last_frame'],
                        'frame_count': len(track_data['frames']),
                        'avg_confidence': np.mean(track_data['confidences'])
                    })
                expired_tracks.append(track_id)
        for track_id in expired_tracks:
            del self.active_tracks[track_id]
    
    def _bbox_distance(self, bbox1: Tuple, bbox2: Tuple) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    def finalize(self):
        for track_data in self.active_tracks.values():
            if len(track_data['frames']) >= MIN_FRAMES_PER_HIT:
                self.all_tracks.append({
                    'start_frame': track_data['start_frame'],
                    'end_frame': track_data['last_frame'],
                    'frame_count': len(track_data['frames']),
                    'avg_confidence': np.mean(track_data['confidences'])
                })
    
    def to_csv(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["start_sec", "end_sec", "duration_sec", 
                             "start_frame", "end_frame", "frame_count", "avg_confidence"])
            for track in self.all_tracks:
                start_t = track['start_frame'] / self.fps
                end_t = track['end_frame'] / self.fps
                writer.writerow([
                    f"{start_t:.3f}", f"{end_t:.3f}", f"{end_t - start_t:.3f}",
                    track['start_frame'], track['end_frame'], 
                    track['frame_count'], f"{track['avg_confidence']:.3f}"
                ])


def main():
    session = init_session()
    ref_embedding = get_reference_embedding(session)
    print("Reference embedding extracted.")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = EnhancedMultiFaceTracker(fps=fps)

    writer = None
    if ANNOTATED_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(ANNOTATED_VIDEO), fourcc, 
                                 min(fps, MAX_DISPLAY_FPS), (width, height))

    pbar = tqdm(total=total_frames, desc="Processing video with all-face annotations")
    frame_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        pbar.update(1)

        if frame_idx % DETECT_EVERY_N_FRAMES != 0:
            detected_matches = []
            all_detections = []
        else:
            detected_matches, all_detections = enhanced_face_detection(session, frame, ref_embedding)

        tracker.update(frame_idx, detected_matches)

        if writer is not None:
            vis_frame = frame.copy()
            for face_data in all_detections:
                x1, y1, x2, y2 = face_data['bbox']
                confidence = face_data['confidence']
                color = (0, 255, 0) if face_data['is_match'] else (255, 0, 0)  # Green for match, blue for other
                label = "MATCH" if face_data['is_match'] else "FACE"
                
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_frame, f"{label} {confidence:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 1)
            
            cv2.putText(vis_frame, f"Frame: {frame_idx}, Faces: {len(all_detections)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(vis_frame)

    tracker.finalize()
    cap.release()
    if writer:
        writer.release()
    pbar.close()

    tracker.to_csv(OUT_CSV)
    print(f"\nProcessing complete!")
    print(f"Total matched tracks found: {len(tracker.all_tracks)}")
    print(f"Results saved to: {OUT_CSV}")
    if ANNOTATED_VIDEO:
        print(f"Annotated video with all faces: {ANNOTATED_VIDEO}")


if __name__ == "__main__":
    main()
