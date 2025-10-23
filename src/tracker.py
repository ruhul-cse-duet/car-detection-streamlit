# src/tracker.py
from typing import List, Dict, Tuple

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter + 1e-6
    return inter / denom

class _Track:
    __slots__ = ("id", "bbox", "misses", "hits")
    def __init__(self, tid: int, bbox: List[int]):
        self.id = tid
        self.bbox = bbox  # [x1,y1,x2,y2]
        self.misses = 0
        self.hits = 1

class CarTracker:
    """
    Very small IoU-based tracker.
    - No Kalman filter, no Hungarian dependency.
    - Greedy matching by IoU (high to low).
    """
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 20):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: List[_Track] = []
        self._next_id = 1

    def reset(self):
        self.tracks.clear()
        self._next_id = 1

    def update(self, det_bboxes: List[List[int]]) -> List[Dict]:
        """
        det_bboxes: list of [x1,y1,x2,y2] for the current frame (cars only).
        Returns: list of dicts [{'id': int, 'bbox': [x1,y1,x2,y2]}]
        """
        # 1) Build IoU pairs (track_idx, det_idx, iou)
        pairs: List[Tuple[int,int,float]] = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(det_bboxes):
                pairs.append((ti, di, iou_xyxy(t.bbox, d)))
        # 2) Sort by IoU desc, greedy assign
        pairs.sort(key=lambda x: x[2], reverse=True)
        used_t = set()
        used_d = set()
        for ti, di, ov in pairs:
            if ov < self.iou_threshold:
                break
            if ti in used_t or di in used_d:
                continue
            # assign
            self.tracks[ti].bbox = det_bboxes[di]
            self.tracks[ti].misses = 0
            self.tracks[ti].hits += 1
            used_t.add(ti)
            used_d.add(di)

        # 3) Unmatched detections -> new tracks
        for di, d in enumerate(det_bboxes):
            if di not in used_d:
                self.tracks.append(_Track(self._next_id, d))
                self._next_id += 1

        # 4) Unmatched tracks -> age/maybe remove
        alive_tracks: List[_Track] = []
        for ti, t in enumerate(self.tracks):
            if ti not in used_t and t.hits > 0:  # not matched this frame
                t.misses += 1
            if t.misses <= self.max_age:
                alive_tracks.append(t)
        self.tracks = alive_tracks

        # 5) Return current tracks
        return [{"id": t.id, "bbox": t.bbox} for t in self.tracks]
