#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import serial.tools.list_ports
from PySide6 import QtCore, QtWidgets

from camera_manager import CameraManager
from fov_calibrator import FovCalibrator
from motion_controller import MotionController
from scan_planner import ScanPlanner

CONFIG_FILE = "aruco_calib.json"
CENTER_X = 54.0
CENTER_Y = 110.0
RESOLUTIONS = ["1920x1080", "2560x1440", "3840x2160"]
QUALITY_LABELS = {"FHD": "1920x1080", "2K": "2560x1440", "4K": "3840x2160"}


def parse_resolution(res):
    w, h = str(res).lower().split("x")
    return int(w), int(h)


class QtLogHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        self.signal.emit(self.format(record))


class ScannerUI(QtWidgets.QMainWindow):
    log_signal = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ArUco scanner")
        self.resize(1120, 760)

        self.logger = logging.getLogger("aruco")
        self.logger.setLevel(logging.INFO)
        self.log_signal.connect(self._append_log)
        handler = QtLogHandler(self.log_signal)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        self.logger.addHandler(handler)

        self.motion = MotionController(self.logger)
        self.camera = CameraManager(self.logger)
        self.calibrator = FovCalibrator(self.logger, marker_size_mm=5.0)
        self.planner = ScanPlanner(area_w=100.0, area_h=100.0)

        self.config = self._load_config()
        self.last_capture_time = 0.12
        self.last_write_time = 0.05

        self._build_ui()
        self._refresh_ports()
        self._refresh_cameras()
        self._sync_from_config()

    def _default_config(self):
        return {
            "marker_size_mm": 5.0,
            "profiles": {},
            "last_z": 83.0,
            "last_resolution": "2560x1440",
            "overlap": 0.20,
        }

    def _load_config(self):
        if not os.path.exists(CONFIG_FILE):
            return self._default_config()
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            out = self._default_config()
            out.update(data)
            if "profiles" not in out:
                out["profiles"] = {}
            return out
        except Exception:
            return self._default_config()

    def _save_config(self):
        with open(CONFIG_FILE, "w", encoding="utf-8") as fp:
            json.dump(self.config, fp, indent=2, ensure_ascii=False)

    def _build_ui(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)

        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)
        self.ports_combo = QtWidgets.QComboBox()
        top.addWidget(QtWidgets.QLabel("Serial"))
        top.addWidget(self.ports_combo)
        b_refresh = QtWidgets.QPushButton("Обновить")
        b_refresh.clicked.connect(self._refresh_ports)
        top.addWidget(b_refresh)
        b_conn = QtWidgets.QPushButton("Подключить")
        b_conn.clicked.connect(self._connect_serial)
        top.addWidget(b_conn)
        top.addWidget(QtWidgets.QLabel("Feed"))
        self.feed_edit = QtWidgets.QLineEdit("1800")
        self.feed_edit.setFixedWidth(80)
        top.addWidget(self.feed_edit)
        top.addStretch(1)

        calib = QtWidgets.QGroupBox("Калибровка")
        root.addWidget(calib)
        cl = QtWidgets.QGridLayout(calib)
        self.z_edit = QtWidgets.QLineEdit("83.0")
        self.z_edit.setFixedWidth(100)
        cl.addWidget(QtWidgets.QLabel("Z (мм)"), 0, 0)
        cl.addWidget(self.z_edit, 0, 1)
        self.cams_combo = QtWidgets.QComboBox()
        cl.addWidget(QtWidgets.QLabel("Камера"), 0, 2)
        cl.addWidget(self.cams_combo, 0, 3)
        self.move_center_btn = QtWidgets.QPushButton("В центр + на Z")
        self.move_center_btn.clicked.connect(self._move_center_z)
        cl.addWidget(self.move_center_btn, 0, 4)
        self.calib_btn = QtWidgets.QPushButton("Калибровать все разрешения")
        self.calib_btn.clicked.connect(self._start_calibration)
        cl.addWidget(self.calib_btn, 0, 5)
        self.calib_status = QtWidgets.QPlainTextEdit()
        self.calib_status.setReadOnly(True)
        self.calib_status.setFixedHeight(110)
        cl.addWidget(self.calib_status, 1, 0, 1, 6)

        scan = QtWidgets.QGroupBox("Сканирование 100×100")
        root.addWidget(scan)
        sl = QtWidgets.QGridLayout(scan)
        self.quality_combo = QtWidgets.QComboBox()
        self.quality_combo.addItems(list(QUALITY_LABELS.keys()))
        self.quality_combo.currentTextChanged.connect(self._update_scan_metrics)
        sl.addWidget(QtWidgets.QLabel("Качество"), 0, 0)
        sl.addWidget(self.quality_combo, 0, 1)

        self.overlap_spin = QtWidgets.QDoubleSpinBox()
        self.overlap_spin.setRange(0, 80)
        self.overlap_spin.setValue(20)
        self.overlap_spin.valueChanged.connect(self._update_scan_metrics)
        sl.addWidget(QtWidgets.QLabel("Overlap %"), 0, 2)
        sl.addWidget(self.overlap_spin, 0, 3)

        self.fov_move_check = QtWidgets.QCheckBox("FOV=Move")
        self.fov_move_check.toggled.connect(self._update_scan_metrics)
        sl.addWidget(self.fov_move_check, 0, 4)

        self.info_label = QtWidgets.QLabel("-")
        sl.addWidget(self.info_label, 1, 0, 1, 5)

        self.scan_btn = QtWidgets.QPushButton("Сканировать")
        self.scan_btn.clicked.connect(self._start_scan)
        sl.addWidget(self.scan_btn, 0, 5)

        self.progress = QtWidgets.QProgressBar()
        root.addWidget(self.progress)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        root.addWidget(self.log_view, 1)

    def _sync_from_config(self):
        self.z_edit.setText(f"{float(self.config.get('last_z', 83.0)):.2f}")
        last_res = self.config.get("last_resolution", "2560x1440")
        for k, v in QUALITY_LABELS.items():
            if v == last_res:
                self.quality_combo.setCurrentText(k)
                break
        self.overlap_spin.setValue(float(self.config.get("overlap", 0.2)) * 100.0)
        self._update_scan_metrics()

    def _refresh_ports(self):
        self.ports_combo.clear()
        for p in serial.tools.list_ports.comports():
            self.ports_combo.addItem(p.device)

    def _refresh_cameras(self):
        self.cams_combo.clear()
        for cam_id, label in self.camera.list_cameras():
            self.cams_combo.addItem(label, cam_id)

    def _connect_serial(self):
        port = self.ports_combo.currentText().strip()
        if not port:
            QtWidgets.QMessageBox.warning(self, "Serial", "Нет порта")
            return
        if self.motion.connect(port):
            self.logger.info("Serial connected: %s", port)

    def _append_log(self, text):
        self.log_view.appendPlainText(text)

    def _active_profile(self):
        z = f"{float(self.z_edit.text()):.2f}"
        res = QUALITY_LABELS[self.quality_combo.currentText()]
        return self.config.get("profiles", {}).get(z, {}).get(res)

    def _update_scan_metrics(self):
        profile = self._active_profile()
        if not profile:
            self.info_label.setText("Нет профиля FOV для текущих Z/качества. Выполните калибровку.")
            return
        fov_x = float(profile["fovX"])
        fov_y = float(profile["fovY"])
        overlap = self.overlap_spin.value() / 100.0
        plan = self.planner.plan(CENTER_X, CENTER_Y, fov_x, fov_y, overlap=overlap, fov_move=self.fov_move_check.isChecked())

        step_len = math.hypot(plan["stepX"], plan["stepY"])
        feed = max(1.0, float(self.feed_edit.text() or "1800"))
        t_move = (step_len / (feed / 60.0))
        t_settle = 0.20
        t_per_frame = t_move + t_settle + self.last_capture_time + self.last_write_time
        total_s = plan["frames"] * t_per_frame
        mins, secs = int(total_s // 60), int(total_s % 60)
        self.info_label.setText(
            f"FOV {fov_x:.2f}×{fov_y:.2f} мм | step {plan['stepX']:.2f}×{plan['stepY']:.2f} мм | "
            f"grid {plan['cols']}×{plan['rows']} ({plan['frames']} кадров) | примерно: {mins} мин {secs} сек"
        )
        self._current_plan = plan

    def _move_center_z(self):
        if not self.motion.connected:
            QtWidgets.QMessageBox.warning(self, "Serial", "Не подключено")
            return
        z = float(self.z_edit.text())
        feed = float(self.feed_edit.text() or 1800)
        ok = self.motion.home()
        ok = ok and self.motion.move_xy(CENTER_X, CENTER_Y, feed)
        ok = ok and self.motion.move_z(z, feed)
        ok = ok and self.motion.wait()
        if ok:
            self.logger.info("Move to center and Z done")
            self.config["last_z"] = z
            self._save_config()

    def _start_calibration(self):
        threading.Thread(target=self._calibrate_all, daemon=True).start()

    def _calibrate_all(self):
        if not self.motion.connected:
            QtWidgets.QMessageBox.warning(self, "Serial", "Не подключено")
            return
        z = float(self.z_edit.text())
        self._move_center_z()
        cam = self.cams_combo.currentData() or "0"

        z_key = f"{z:.2f}"
        self.config["profiles"].setdefault(z_key, {})
        lines = []
        for res in RESOLUTIONS:
            result = self.calibrator.calibrate(self.camera, cam, res)
            if result is None:
                lines.append(f"{res}: FAIL (проверь фокус/освещение)")
                continue
            profile = {
                "fovX": round(result["fovX_mm"], 4),
                "fovY": round(result["fovY_mm"], 4),
                "mm_per_px": round(result["mm_per_px"], 6),
                "dict": result["dict_used"],
                "quality": round(result["confidence"], 2),
                "success_frames": int(result["success_frames"]),
                "z": z,
                "res": res,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            self.config["profiles"][z_key][res] = profile
            lines.append(f"{res}: OK FOV={profile['fovX']:.2f}×{profile['fovY']:.2f} мм")

        self.config["last_z"] = z
        self.config["last_resolution"] = QUALITY_LABELS[self.quality_combo.currentText()]
        self.config["overlap"] = self.overlap_spin.value() / 100.0
        self._save_config()
        self.calib_status.setPlainText("\n".join(lines))
        self._update_scan_metrics()

    def _start_scan(self):
        threading.Thread(target=self._scan, daemon=True).start()

    def _scan(self):
        if not self.motion.connected:
            QtWidgets.QMessageBox.warning(self, "Serial", "Не подключено")
            return
        profile = self._active_profile()
        if not profile:
            QtWidgets.QMessageBox.warning(self, "Scan", "Нет профиля. Калибруйте")
            return

        fov_x = float(profile["fovX"])
        fov_y = float(profile["fovY"])
        overlap = self.overlap_spin.value() / 100.0
        plan = self.planner.plan(CENTER_X, CENTER_Y, fov_x, fov_y, overlap=overlap, fov_move=self.fov_move_check.isChecked())
        feed = float(self.feed_edit.text() or 1800)
        z = float(self.z_edit.text())
        res = QUALITY_LABELS[self.quality_combo.currentText()]
        w, h = parse_resolution(res)
        cam = self.cams_combo.currentData() or "0"

        base = Path("scans")
        base.mkdir(exist_ok=True)
        out = base / f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        frames_dir = out / "scan_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        self.motion.home()
        self.motion.move_xy(CENTER_X, CENTER_Y, feed)
        self.motion.move_z(z, feed)
        self.motion.wait()

        shots = []
        self.progress.setValue(0)
        total = len(plan["positions"])
        for i, (col, row, x, y) in enumerate(plan["positions"]):
            if not self.motion.move_xy(x, y, feed):
                continue
            self.motion.wait()
            time.sleep(0.2)
            t0 = time.perf_counter()
            frame = self.camera.snap(cam, w, h)
            self.last_capture_time = time.perf_counter() - t0
            if frame is None:
                continue
            fn = f"frame_{i:04d}_c{col}_r{row}.png"
            t1 = time.perf_counter()
            cv2.imwrite(str(frames_dir / fn), frame)
            self.last_write_time = time.perf_counter() - t1
            shots.append({"index": i, "col": col, "row": row, "x": x, "y": y, "file": f"scan_frames/{fn}"})
            self.progress.setValue(int((i + 1) * 100 / total))

        payload = {
            "area": {"width": 100.0, "height": 100.0},
            "resolution": res,
            "fovX": fov_x,
            "fovY": fov_y,
            "stepX": plan["stepX"],
            "stepY": plan["stepY"],
            "cols": plan["cols"],
            "rows": plan["rows"],
            "frames_total": plan["frames"],
            "shots": shots,
        }
        with open(out / "shots.json", "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)

        self.logger.info("Scan finished: %s", out)

    def closeEvent(self, event):
        self._save_config()
        self.camera.close()
        self.motion.disconnect()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = ScannerUI()
    window.show()
    app.exec()
