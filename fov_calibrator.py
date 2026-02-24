import statistics
import time

import cv2
import numpy as np


def parse_resolution(res):
    w, h = str(res).lower().split("x")
    return int(w), int(h)


class FovCalibrator:
    PREPROCESSORS = (
        "gray",
        "clahe",
        "gauss",
        "sharpen",
        "adaptive_thresh",
    )

    def __init__(self, logger, marker_size_mm=5.0, min_success=3, frames_count=8):
        self.logger = logger
        self.marker_size_mm = marker_size_mm
        self.min_success = min_success
        self.frames_count = frames_count
        self.dict_map = self._build_dictionary_map()

    def _build_dictionary_map(self):
        result = {}
        for name in dir(cv2.aruco):
            if name.startswith("DICT_"):
                value = getattr(cv2.aruco, name)
                if isinstance(value, int):
                    result[name] = cv2.aruco.getPredefinedDictionary(value)
        return result

    def _preprocess(self, gray, mode):
        if mode == "gray":
            return gray
        if mode == "clahe":
            return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        if mode == "gauss":
            return cv2.GaussianBlur(gray, (5, 5), 0)
        if mode == "sharpen":
            blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
            return cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
        if mode == "adaptive_thresh":
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        return gray

    def _detect_best(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best = None
        params = cv2.aruco.DetectorParameters()
        for dict_name, dictionary in self.dict_map.items():
            for prep in self.PREPROCESSORS:
                processed = self._preprocess(gray, prep)
                corners, ids, _ = cv2.aruco.detectMarkers(processed, dictionary, parameters=params)
                if ids is None or len(ids) == 0:
                    continue
                perimeters = [cv2.arcLength(c.reshape(-1, 1, 2).astype(np.float32), True) for c in corners]
                quality = float(sum(perimeters) + 100.0 * len(ids))
                if not best or quality > best["quality"]:
                    best = {
                        "dict": dict_name,
                        "prep": prep,
                        "corners": corners,
                        "ids": ids,
                        "quality": quality,
                    }
        return best

    def calibrate(self, camera_manager, cam_id, resolution, exposure_us=None):
        w, h = parse_resolution(resolution)
        if exposure_us:
            camera_manager.set_exposure(cam_id, exposure_us, (w, h))
        camera_manager.warm_up(cam_id, w, h)

        mm_per_px_samples = []
        qualities = []
        dict_votes = {}
        success_frames = 0

        for _ in range(self.frames_count):
            frame = camera_manager.snap(cam_id, w, h)
            if frame is None:
                time.sleep(0.1)
                continue
            best = self._detect_best(frame)
            if not best:
                continue
            for corner in best["corners"]:
                pts = corner.reshape(4, 2)
                side = float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]))
                if side > 0:
                    mm_per_px_samples.append(self.marker_size_mm / side)
            success_frames += 1
            qualities.append(best["quality"])
            dict_votes[best["dict"]] = dict_votes.get(best["dict"], 0) + 1
            time.sleep(0.05)

        if success_frames < self.min_success or not mm_per_px_samples:
            return None

        mm_per_px = float(statistics.median(mm_per_px_samples))
        dict_used = max(dict_votes, key=dict_votes.get)
        fov_x = w * mm_per_px
        fov_y = h * mm_per_px
        return {
            "fovX_mm": float(fov_x),
            "fovY_mm": float(fov_y),
            "mm_per_px": mm_per_px,
            "dict_used": dict_used,
            "confidence": float(statistics.mean(qualities)) if qualities else 0.0,
            "success_frames": success_frames,
        }
