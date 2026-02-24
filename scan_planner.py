import math


class ScanPlanner:
    def __init__(self, area_w=100.0, area_h=100.0):
        self.area_w = area_w
        self.area_h = area_h

    def compute_steps(self, fov_x, fov_y, overlap=0.2, fov_move=False):
        if fov_move:
            return fov_x, fov_y
        factor = 1.0 - max(0.0, min(0.95, overlap))
        return fov_x * factor, fov_y * factor

    def plan(self, base_x, base_y, fov_x, fov_y, overlap=0.2, fov_move=False):
        step_x, step_y = self.compute_steps(fov_x, fov_y, overlap, fov_move)
        cols = int(math.ceil(self.area_w / step_x)) + 1
        rows = int(math.ceil(self.area_h / step_y)) + 1

        positions = []
        for row in range(rows):
            y = base_y + min(self.area_h, row * step_y)
            seq = range(cols) if row % 2 == 0 else reversed(range(cols))
            for col in seq:
                x = base_x + min(self.area_w, col * step_x)
                positions.append((col, row, x, y))
        return {
            "stepX": step_x,
            "stepY": step_y,
            "cols": cols,
            "rows": rows,
            "frames": cols * rows,
            "positions": positions,
        }
