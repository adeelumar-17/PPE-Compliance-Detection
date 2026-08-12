"""
PPE Compliance Detection Engine
================================
Core detection, tracking, and compliance logic for construction-site PPE monitoring.
Extracted and refined from the YOLO11m fine-tuning notebook.

Classes detected:
    0: helmet      — person wearing a helmet
    1: no-helmet   — person without a helmet
    2: no-vest     — person without a safety vest
    3: person      — detected person
    4: vest        — person wearing a safety vest
"""

import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = {0: "helmet", 1: "no-helmet", 2: "no-vest", 3: "person", 4: "vest"}
CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}

PERSON_CLASS = 3
HELMET_CLASS = 0
NO_HELMET_CLASS = 1
VEST_CLASS = 4
NO_VEST_CLASS = 2

# Default thresholds
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.50
DEFAULT_IMGSZ = 640

# Colors (BGR for OpenCV)
COLOR_COMPLIANT = (0, 200, 0)       # Green
COLOR_VIOLATION = (0, 0, 255)       # Red
COLOR_UNKNOWN = (255, 255, 255)     # White
COLOR_PPE_BOX = (200, 200, 0)      # Cyan-ish
COLOR_PERSON_BOX = (255, 180, 0)   # Orange-ish

# Colors (RGB for Streamlit / PIL display)
RGB_COMPLIANT = (0, 200, 0)
RGB_VIOLATION = (255, 40, 40)
RGB_UNKNOWN = (200, 200, 200)


# ──────────────────────────────────────────────────────────────────────────────
# PPE Detector — wraps YOLO model
# ──────────────────────────────────────────────────────────────────────────────

class PPEDetector:
    """Wraps a YOLO model for PPE detection with structured output."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame: np.ndarray, conf: float = DEFAULT_CONF,
               iou: float = DEFAULT_IOU, imgsz: int = DEFAULT_IMGSZ) -> list[dict]:
        """
        Run inference on a single frame.

        Returns a list of detection dicts:
            {"class_id": int, "class_name": str, "confidence": float, "box": [x1, y1, x2, y2]}
        """
        result = self.model.predict(
            source=frame, imgsz=imgsz, conf=conf, iou=iou,
            device=self.device, verbose=False
        )[0]

        detections = []
        if result.boxes is not None:
            for box, cls, confidence in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy(),
            ):
                detections.append({
                    "class_id": int(cls),
                    "class_name": CLASS_NAMES.get(int(cls), f"class_{cls}"),
                    "confidence": float(confidence),
                    "box": box.tolist(),
                })
        return detections

    def split_detections(self, detections: list[dict]):
        """Split detections into persons and PPE items."""
        persons = [d for d in detections if d["class_id"] == PERSON_CLASS]
        ppe = [d for d in detections if d["class_id"] != PERSON_CLASS]
        return persons, ppe


# ──────────────────────────────────────────────────────────────────────────────
# PPE ↔ Person Association & Compliance
# ──────────────────────────────────────────────────────────────────────────────

def box_center(box):
    """Return (cx, cy) of an [x1, y1, x2, y2] box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_inside_box(point, box):
    """Check if a (px, py) point lies inside an [x1, y1, x2, y2] box."""
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def associate_ppe_to_person(person_box, ppe_detections):
    """
    Associate helmet/vest detections with a single person using anatomical regions.

    Head region  = top 40% of person box  → helmet / no-helmet
    Torso region = 20%-80% of person box  → vest / no-vest

    Returns dict with keys: "helmet", "no-helmet", "vest", "no-vest"
    Each value is the highest-confidence matching detection, or None.
    """
    px1, py1, px2, py2 = person_box
    person_height = py2 - py1

    head_region = (px1, py1, px2, py1 + person_height * 0.40)
    torso_region = (px1, py1 + person_height * 0.20, px2, py1 + person_height * 0.80)

    best = {"helmet": None, "no-helmet": None, "vest": None, "no-vest": None}
    region_for = {
        HELMET_CLASS: (head_region, "helmet"),
        NO_HELMET_CLASS: (head_region, "no-helmet"),
        VEST_CLASS: (torso_region, "vest"),
        NO_VEST_CLASS: (torso_region, "no-vest"),
    }

    for det in ppe_detections:
        region_info = region_for.get(det["class_id"])
        if region_info is None:
            continue
        region, key = region_info
        if point_inside_box(box_center(det["box"]), region):
            if best[key] is None or det["confidence"] > best[key]["confidence"]:
                best[key] = det
    return best


def determine_compliance(associated):
    """
    Determine PPE compliance from associated PPE detections.

    Returns:
        {"helmet": str, "vest": str, "compliance": str}
        where compliance ∈ {"COMPLIANT", "VIOLATION", "UNKNOWN"}
    """
    helmet_status = (
        "helmet" if associated["helmet"]
        else ("no-helmet" if associated["no-helmet"] else "unknown")
    )
    vest_status = (
        "vest" if associated["vest"]
        else ("no-vest" if associated["no-vest"] else "unknown")
    )

    if helmet_status == "helmet" and vest_status == "vest":
        compliance = "COMPLIANT"
    elif helmet_status == "no-helmet" or vest_status == "no-vest":
        compliance = "VIOLATION"
    else:
        compliance = "UNKNOWN"

    return {"helmet": helmet_status, "vest": vest_status, "compliance": compliance}


def analyze_frame(detector, frame, conf=DEFAULT_CONF, iou=DEFAULT_IOU, imgsz=DEFAULT_IMGSZ):
    """
    Full single-frame analysis pipeline.

    Returns:
        (annotated_frame, person_results, summary)
    where person_results is a list of dicts per person, and summary is aggregate stats.
    """
    detections = detector.detect(frame, conf=conf, iou=iou, imgsz=imgsz)
    persons, ppe = detector.split_detections(detections)

    person_results = []
    for i, person in enumerate(persons, start=1):
        associated = associate_ppe_to_person(person["box"], ppe)
        compliance = determine_compliance(associated)
        person_results.append({
            "id": i,
            "box": person["box"],
            "confidence": person["confidence"],
            "helmet_status": compliance["helmet"],
            "vest_status": compliance["vest"],
            "compliance": compliance["compliance"],
            "associated_ppe": associated,
        })

    # Build summary
    total = len(person_results)
    compliant = sum(1 for p in person_results if p["compliance"] == "COMPLIANT")
    violations = sum(1 for p in person_results if p["compliance"] == "VIOLATION")
    unknown = sum(1 for p in person_results if p["compliance"] == "UNKNOWN")

    summary = {
        "total_persons": total,
        "compliant": compliant,
        "violations": violations,
        "unknown": unknown,
        "all_detections": detections,
        "ppe_detections": ppe,
    }

    # Annotate frame
    annotated = annotate_frame(frame.copy(), person_results, ppe)

    return annotated, person_results, summary


# ──────────────────────────────────────────────────────────────────────────────
# Person Tracker — IoU-based frame-to-frame tracking
# ──────────────────────────────────────────────────────────────────────────────

def _iou(box_a, box_b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class PersonTracker:
    """
    Simple IoU-based multi-object tracker for persons.

    Matches current-frame detections to previous-frame tracks by IoU.
    Unmatched detections get new IDs. Tracks that go unmatched for
    `max_lost` frames are dropped.
    """

    def __init__(self, iou_threshold=0.3, max_lost=15):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks = {}   # track_id -> {"box": [...], "lost": int}

    def update(self, boxes):
        """
        Update tracker with new frame's person bounding boxes.

        Args:
            boxes: list of [x1, y1, x2, y2] boxes

        Returns:
            list of (track_id, box) tuples in same order as input boxes
        """
        if not boxes:
            # Age out all tracks
            for tid in list(self.tracks):
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
            return []

        if not self.tracks:
            # First frame or all tracks lost — assign fresh IDs
            results = []
            for box in boxes:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": box, "lost": 0}
                results.append((tid, box))
            return results

        # Compute IoU matrix: tracks × detections
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]["box"] for tid in track_ids]

        iou_matrix = np.zeros((len(track_boxes), len(boxes)))
        for i, tbox in enumerate(track_boxes):
            for j, dbox in enumerate(boxes):
                iou_matrix[i, j] = _iou(tbox, dbox)

        # Greedy matching (highest IoU first)
        matched_tracks = set()
        matched_dets = set()
        assignments = {}  # det_index -> track_id

        flat_indices = np.argsort(-iou_matrix.ravel())
        for flat_idx in flat_indices:
            i, j = divmod(int(flat_idx), len(boxes))
            if iou_matrix[i, j] < self.iou_threshold:
                break
            if i in matched_tracks or j in matched_dets:
                continue
            tid = track_ids[i]
            assignments[j] = tid
            self.tracks[tid] = {"box": boxes[j], "lost": 0}
            matched_tracks.add(i)
            matched_dets.add(j)

        # Unmatched detections → new tracks
        for j, box in enumerate(boxes):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": box, "lost": 0}
                assignments[j] = tid

        # Unmatched tracks → age out
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return [(assignments[j], boxes[j]) for j in range(len(boxes))]

    def reset(self):
        """Reset tracker state."""
        self.next_id = 1
        self.tracks = {}


# ──────────────────────────────────────────────────────────────────────────────
# Violation Event Log — temporal confirmation
# ──────────────────────────────────────────────────────────────────────────────

class ViolationEventLog:
    """
    Tracks violation events with temporal confirmation.

    A violation must persist for `confirmation_frames` consecutive frames before
    it is promoted to a confirmed event. The event stays open until
    `end_frames` consecutive clear frames pass.
    """

    def __init__(self, confirmation_frames=5, end_frames=10):
        self.confirmation_frames = confirmation_frames
        self.end_frames = end_frames

        self.events = []
        self.next_event_id = 1

        # Running state
        self.active = False
        self.violation_type = None
        self.violation_conf = 0.0
        self._consecutive_violation = 0
        self._consecutive_clear = 0
        self._pending_type = None
        self._pending_conf = 0.0
        self._event_start_frame = 0

    def update(self, frame_number, fps, has_violation, violation_type=None, confidence=0.0):
        """Process one frame's violation status."""
        if has_violation:
            self._consecutive_clear = 0
            self._consecutive_violation += 1
            self._pending_type = violation_type
            self._pending_conf = max(self._pending_conf, confidence)

            # Promote to active event after confirmation threshold
            if not self.active and self._consecutive_violation >= self.confirmation_frames:
                self.active = True
                self.violation_type = self._pending_type
                self.violation_conf = self._pending_conf
                self._event_start_frame = frame_number - self.confirmation_frames + 1
        else:
            self._consecutive_violation = 0
            self._pending_conf = 0.0

            if self.active:
                self._consecutive_clear += 1
                if self._consecutive_clear >= self.end_frames:
                    self._close_event(frame_number, fps)

    def _close_event(self, frame_number, fps):
        """Close the currently active event."""
        if not self.active:
            return
        event = {
            "event_id": self.next_event_id,
            "violation": self.violation_type or "Unknown",
            "confidence": self.violation_conf,
            "start_frame": self._event_start_frame,
            "end_frame": frame_number,
            "start_time": f"{self._event_start_frame / fps:.1f}s" if fps > 0 else "N/A",
            "end_time": f"{frame_number / fps:.1f}s" if fps > 0 else "N/A",
        }
        self.events.append(event)
        self.next_event_id += 1
        self.active = False
        self.violation_type = None
        self.violation_conf = 0.0
        self._consecutive_clear = 0

    def close_if_active(self, frame_number, fps):
        """Force-close any active event (e.g. at end of video)."""
        self._close_event(frame_number, fps)

    def reset(self):
        """Reset all state."""
        self.__init__(self.confirmation_frames, self.end_frames)


# ──────────────────────────────────────────────────────────────────────────────
# Drawing / annotation helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_box_cv2(frame, box, label, color=(255, 255, 255), thickness=2):
    """Draw a bounding box with label on an OpenCV BGR frame."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(y1 - th - 10, 0)), (x1 + tw + 4, max(y1 - 2, 0)), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, max(y1 - 6, 16)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA
    )


def annotate_frame(frame, person_results, ppe_detections):
    """
    Annotate a frame with all detections, compliance status, and summary overlay.

    Args:
        frame: BGR numpy array (will be modified in-place)
        person_results: list of person result dicts from analyze_frame
        ppe_detections: list of PPE detection dicts
    """
    h, w = frame.shape[:2]

    # Draw PPE detections
    for det in ppe_detections:
        cls_name = det["class_name"]
        if cls_name in ("no-helmet", "no-vest"):
            color = COLOR_VIOLATION
        elif cls_name in ("helmet", "vest"):
            color = COLOR_COMPLIANT
        else:
            color = COLOR_PPE_BOX
        draw_box_cv2(frame, det["box"], f"{cls_name} {det['confidence']:.2f}", color)

    # Draw persons with compliance status
    for person in person_results:
        compliance = person["compliance"]
        color = {
            "COMPLIANT": COLOR_COMPLIANT,
            "VIOLATION": COLOR_VIOLATION,
        }.get(compliance, COLOR_UNKNOWN)

        draw_box_cv2(frame, person["box"], f"Person {person['id']}", COLOR_PERSON_BOX)

        # Compliance label below person box
        x1, y1, x2, y2 = map(int, person["box"])
        status_text = f"{compliance}"
        detail_text = f"H:{person['helmet_status']} V:{person['vest_status']}"

        cv2.putText(
            frame, status_text, (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, detail_text, (x1, y2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA
        )

    # Summary overlay (top-right)
    total = len(person_results)
    compliant = sum(1 for p in person_results if p["compliance"] == "COMPLIANT")
    violations = sum(1 for p in person_results if p["compliance"] == "VIOLATION")

    overlay_lines = [
        f"Persons: {total}",
        f"Compliant: {compliant}",
        f"Violations: {violations}",
    ]
    y_offset = 30
    for line in overlay_lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x_pos = w - tw - 15
        cv2.rectangle(frame, (x_pos - 5, y_offset - th - 5), (w - 5, y_offset + 5), (0, 0, 0), -1)
        line_color = COLOR_COMPLIANT if "Compliant" in line else (COLOR_VIOLATION if "Violation" in line else (255, 255, 255))
        cv2.putText(
            frame, line, (x_pos, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2, cv2.LINE_AA
        )
        y_offset += 30

    return frame


def annotate_frame_with_tracking(frame, detector, tracker, event_log,
                                  frame_number, fps, conf, iou, imgsz):
    """
    Full tracked + temporally-confirmed annotation for video/realtime pipelines.
    Returns (annotated_frame, person_results, summary).
    """
    detections = detector.detect(frame, conf=conf, iou=iou, imgsz=imgsz)
    persons_raw, ppe = detector.split_detections(detections)

    tracked = tracker.update([p["box"] for p in persons_raw])

    person_results = []
    frame_has_violation = False
    violation_type = None
    violation_conf = 0.0

    for (track_id, box), person in zip(tracked, persons_raw):
        associated = associate_ppe_to_person(box, ppe)
        compliance = determine_compliance(associated)

        person_results.append({
            "id": track_id,
            "box": box,
            "confidence": person["confidence"],
            "helmet_status": compliance["helmet"],
            "vest_status": compliance["vest"],
            "compliance": compliance["compliance"],
        })

        if compliance["compliance"] == "VIOLATION":
            frame_has_violation = True
            if associated["no-helmet"]:
                violation_type = "No Helmet"
                violation_conf = associated["no-helmet"]["confidence"]
            elif associated["no-vest"]:
                violation_type = "No Vest"
                violation_conf = associated["no-vest"]["confidence"]

    # Update event log
    event_log.update(frame_number, fps, frame_has_violation, violation_type, violation_conf)

    # Annotate
    annotated = annotate_frame(frame.copy(), person_results, ppe)

    # Violation alert banner
    if event_log.active:
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 50), (0, 0, 180), -1)
        cv2.putText(
            annotated, f"⚠ ACTIVE VIOLATION: {event_log.violation_type}",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
        )

    # Summary
    total = len(person_results)
    compliant_count = sum(1 for p in person_results if p["compliance"] == "COMPLIANT")
    violations_count = sum(1 for p in person_results if p["compliance"] == "VIOLATION")

    summary = {
        "total_persons": total,
        "compliant": compliant_count,
        "violations": violations_count,
        "unknown": total - compliant_count - violations_count,
        "active_violation": event_log.active,
        "violation_events": len(event_log.events),
    }

    return annotated, person_results, summary
