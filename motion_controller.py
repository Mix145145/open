import time
import serial


class MotionController:
    def __init__(self, logger, command_timeout_s=5.0):
        self.logger = logger
        self.command_timeout_s = command_timeout_s
        self.ser = None

    @property
    def connected(self):
        return bool(self.ser and self.ser.is_open)

    def connect(self, port, baudrate=115200):
        self.disconnect()
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(1.5)
        return True

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None

    def _readline_until_timeout(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        lines = []
        while time.monotonic() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
            if line:
                lines.append(line)
                if line.lower() == "ok" or line.startswith("ok"):
                    return True, lines
                if "error" in line.lower():
                    return False, lines
        return False, lines

    def _g(self, cmd, timeout_s=None):
        if not self.connected:
            return False
        timeout_s = timeout_s or self.command_timeout_s
        try:
            self.ser.write((cmd + "\n").encode("ascii", errors="ignore"))
            self.ser.flush()
            ok, lines = self._readline_until_timeout(timeout_s)
            if not ok:
                self.logger.info("G-code timeout/error for '%s'. Response: %s", cmd, lines)
            return ok
        except Exception as exc:
            self.logger.info("G-code send failed: %s", exc)
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass
            return False

    def home(self):
        return self._g("G90") and self._g("G28") and self.wait()

    def move_abs(self, x, y, z, f):
        return self._g(f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f} F{int(f)}")

    def move_xy(self, x, y, f):
        return self._g(f"G1 X{x:.2f} Y{y:.2f} F{int(f)}")

    def move_z(self, z, f):
        return self._g(f"G1 Z{z:.2f} F{int(f)}")

    def wait(self):
        return self._g("M400")
