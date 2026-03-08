import numpy as np


class LaneFixer:
    """
    Detects whether detected lanes are all on one side of the frame, and if so,
    synthesises the missing side by translating the fitted quadratic curve by
    the supplied lane width.
    """

    def __init__(self, lane_width=None, samples=20, threshold=1.1):
        self.last_width = lane_width
        self.lane_width = lane_width
        self.threshold = threshold
        self.samples = samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fix(self, lanes, frame_width, n_samples=20):
        """
        Check whether *lanes* cover only the left or only the right half of the
        frame, and if so synthesise the missing side via a polynomial shift.
        Returns:
            The (possibly extended) list of lanes.  The original lanes are
            kept as-is; only a synthetic lane is appended when needed.
        """
        if not lanes:
            return lanes

        cx = frame_width / 2.0
        left_miss, right_miss = self._check_missing_sides(lanes, cx)

        # Both sides present — auto-update width and return as-is
        if not left_miss and not right_miss:
            left_lanes = [l for l in lanes if self._mean_x(l) <= cx]
            right_lanes = [l for l in lanes if self._mean_x(l) > cx]
            left_x = np.mean([self._mean_x(l) for l in left_lanes])
            right_x = np.mean([self._mean_x(l) for l in right_lanes])
            self.update(new_width=float(abs(right_x - left_x)))
            return lanes

        # One side missing — need last_width to synthesise
        if self.last_width is None:
            return lanes

        result = list(lanes)  # shallow copy – originals are untouched

        if right_miss and not left_miss:
            anchor_lanes = [l for l in lanes if self._mean_x(l) <= cx]
            synthetic = self._synthesise(anchor_lanes, shift=+self.last_width, n_samples=n_samples)
            if synthetic:
                result.append(synthetic)

        elif left_miss and not right_miss:
            anchor_lanes = [l for l in lanes if self._mean_x(l) > cx]
            synthetic = self._synthesise(anchor_lanes, shift=-self.last_width, n_samples=n_samples)
            if synthetic:
                result.insert(0, synthetic)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def update(self, new_width=None):
        """
        Update last_width based on a new observation.

        Accepts the update only when the new width is within the ratio range
        [last_width / threshold, last_width * threshold], i.e. the change
        does not exceed ±(threshold-1)*100 %.

        If new_width is not provided, falls back to self.lane_width.
        """
        w = new_width if new_width is not None else self.lane_width
        if w is None:
            return
        if self.last_width is None:
            # First valid observation — accept unconditionally
            self.last_width = w
        else:
            lo = self.last_width / self.threshold
            hi = self.last_width * self.threshold
            if lo <= w <= hi:
                self.last_width = w
            # else: change is too large (outlier) — silently ignore

    @staticmethod
    def _mean_x(lane):
        return sum(p[0] for p in lane) / len(lane)

    def _check_missing_sides(self, lanes, cx):
        """Return (left_missing, right_missing) booleans."""
        has_left = any(self._mean_x(l) <= cx for l in lanes)
        has_right = any(self._mean_x(l) > cx for l in lanes)
        return (not has_left), (not has_right)

    def _fit_poly(self, lane):
        """
        Fit  x = f(y)  as a quadratic polynomial (degree 2).
        Returns numpy coefficient array (highest degree first).
        """
        pts = np.asarray(lane, dtype=float)
        x = pts[:, 0]
        y = pts[:, 1]
        order = np.argsort(y)
        return np.polyfit(y[order], x[order], deg=2)

    def _synthesise(self, anchor_lanes, shift, n_samples):
        """
        Fit a single quadratic through *all* anchor lane points combined,
        then shift it horizontally by *shift* pixels and sample *n_samples*
        points over the observed y range.
        """
        all_pts = [p for lane in anchor_lanes for p in lane]
        if len(all_pts) < 3:  # minimum for degree-2 fit
            return None

        coef = self._fit_poly(all_pts)

        all_y = [p[1] for p in all_pts]
        y_lo, y_hi = min(all_y), max(all_y)
        if y_lo >= y_hi:
            return None

        ys = np.linspace(y_lo, y_hi, n_samples)
        xs = np.polyval(coef, ys) + shift

        synthetic = [(int(round(float(x))), int(round(float(y)))) for x, y in zip(xs, ys)]
        return synthetic

