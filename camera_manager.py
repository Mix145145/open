import importlib
import os
import threading
import time

import cv2


DEFAULT_RESOLUTION = (1920, 1080)


def get_picamera2_class():
    if importlib.util.find_spec("picamera2") is None:
        return None
    return importlib.import_module("picamera2").Picamera2


class CameraManager:
    def __init__(self, logger):
        self.logger = logger
        self.picamera2_cls = get_picamera2_class()
        self.picam = None
        self.picam_id = None
        self.picam_resolution = None
        self.lock = threading.Lock()

    def list_cameras(self):
        if self.picamera2_cls:
            infos = self.picamera2_cls.global_camera_info()
            return [(str(i), f"Camera {i}: {info.get('Model', f'Camera {i}')}") for i, info in enumerate(infos)] or [("0", "Camera 0")]
        devices = []
        for idx in range(10):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                devices.append((str(idx), f"/dev/video{idx}"))
            cap.release()
        return devices or [("0", "/dev/video0")]

    def _ensure_picam(self, cam_id, resolution):
        if self.picam and self.picam_id != cam_id:
            self.picam.close()
            self.picam = None
        if not self.picam:
            self.picam = self.picamera2_cls(int(cam_id))
            self.picam_id = cam_id
            self.picam_resolution = None
        if self.picam_resolution != resolution:
            try:
                self.picam.stop()
            except Exception:
                pass
            config = self.picam.create_still_configuration(main={"size": resolution})
            self.picam.configure(config)
            self.picam.start()
            self.picam_resolution = resolution
            time.sleep(0.15)

    def warm_up(self, cam_id, w, h):
        with self.lock:
            if self.picamera2_cls:
                try:
                    self._ensure_picam(cam_id, (w, h))
                    return True
                except Exception as exc:
                    self.logger.info("Picamera warmup failed: %s", exc)
                    return False
            cap = cv2.VideoCapture(int(cam_id) if str(cam_id).isdigit() else cam_id, cv2.CAP_V4L2)
            if not cap.isOpened():
                return False
            cap.set(3, w)
            cap.set(4, h)
            cap.read()
            cap.release()
            return True

    def snap(self, cam_id, w, h):
        with self.lock:
            if self.picamera2_cls:
                try:
                    self._ensure_picam(cam_id, (w, h))
                    frame = self.picam.capture_array()
                    if frame is None:
                        return None
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 and frame.shape[2] >= 3 else frame
                except Exception as exc:
                    self.logger.info("Picamera snap failed: %s", exc)
                    return None
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
            cap = cv2.VideoCapture(int(cam_id) if str(cam_id).isdigit() else cam_id, backend)
            cap.set(3, w)
            cap.set(4, h)
            for _ in range(3):
                cap.grab()
            ok, frame = cap.read()
            cap.release()
            return frame if ok else None

    def set_exposure(self, cam_id, us, resolution):
        us = max(1, int(us))
        with self.lock:
            if self.picamera2_cls:
                try:
                    self._ensure_picam(cam_id, resolution)
                    self.picam.set_controls({"ExposureTime": us})
                    return True
                except Exception as exc:
                    self.logger.info("Picamera exposure failed: %s", exc)
                    return False
            cap = cv2.VideoCapture(int(cam_id) if str(cam_id).isdigit() else cam_id, cv2.CAP_V4L2)
            if not cap.isOpened():
                return False
            cap.set(cv2.CAP_PROP_EXPOSURE, float(us))
            cap.release()
            return True

    def close(self):
        with self.lock:
            if self.picam:
                try:
                    self.picam.stop()
                except Exception:
                    pass
                self.picam.close()
                self.picam = None
