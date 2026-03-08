import numpy as np


class LaneFixer:
    """
    Detects whether detected lanes are all on one side of the frame, and if so,
    synthesises the missing side by translating the fitted quadratic curve by
    the supplied lane width.

    Parameters
    ----------
    lane_width : int | float
        Expected pixel distance between the left and right lane boundaries.
        Used as the horizontal translation offset when generating the missing lane.
    poly_degree : int
        Degree of the polynomial used to fit each lane  (default 2 → quadratic).
    min_points : int
        Minimum number of points a lane must have to be considered valid.
    """

    def __init__(self, lane_width, poly_degree=2, min_points=3):
        self.lane_width = lane_width
        self.poly_degree = poly_degree
        self.min_points = min_points

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fix(self, lanes, frame_width, n_samples=20):
        """
        Check whether *lanes* cover only the left or only the right half of the
        frame, and if so synthesise the missing side via a polynomial shift.

        Parameters
        ----------
        lanes : list[list[tuple[int,int]]]
            Each element is a lane represented as a list of ``(x, y)`` pixel
            tuples, sorted in any y-order.
        frame_width : int
            Width of the video frame in pixels.
        n_samples : int
            Number of evenly-spaced y points to sample when building the
            synthetic lane.

        Returns
        -------
        list[list[tuple[int,int]]]
            The (possibly extended) list of lanes.  The original lanes are
            kept as-is; only a synthetic lane is appended when needed.
        """
        valid = [l for l in lanes if len(l) >= self.min_points]
        if not valid:
            return lanes

        cx = frame_width / 2.0
        left_miss, right_miss = self._check_missing_sides(valid, cx)

        if not left_miss and not right_miss:
            return lanes

        result = list(lanes)  # shallow copy – originals are untouched

        # Choose the "anchor" side that IS present and synthesise the other.
        # When both sides are strangely missing (shouldn't happen after the
        # guard above) we still do nothing.
        if right_miss and not left_miss:
            # All lanes are on the left → synthesise a right-side lane
            anchor_lanes = [l for l in valid if self._mean_x(l) <= cx]
            synthetic = self._synthesise(anchor_lanes, shift=+self.lane_width, n_samples=n_samples)
            if synthetic:
                result.append(synthetic)

        elif left_miss and not right_miss:
            # All lanes are on the right → synthesise a left-side lane
            anchor_lanes = [l for l in valid if self._mean_x(l) > cx]
            synthetic = self._synthesise(anchor_lanes, shift=-self.lane_width, n_samples=n_samples)
            if synthetic:
                result.insert(0, synthetic)  # keep left-to-right order

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        Fit  x = f(y)  as a polynomial of degree *self.poly_degree*.
        Returns numpy coefficient array (highest degree first).
        """
        pts = np.asarray(lane, dtype=float)
        x = pts[:, 0]
        y = pts[:, 1]
        order = np.argsort(y)
        return np.polyfit(y[order], x[order], deg=self.poly_degree)

    def _synthesise(self, anchor_lanes, shift, n_samples):
        """
        Fit a single quadratic through *all* anchor lane points combined,
        then shift it horizontally by *shift* pixels and sample *n_samples*
        points over the observed y range.

        Parameters
        ----------
        anchor_lanes : list[list[tuple]]
            One or more lanes on the present side.
        shift : float
            Pixels to add to every x value (positive → right, negative → left).
        n_samples : int
            Number of points in the returned synthetic lane.

        Returns
        -------
        list[tuple[int,int]] | None
        """
        # Pool all points from the anchor lanes together for a robust fit
        all_pts = [p for lane in anchor_lanes for p in lane]
        if len(all_pts) < self.poly_degree + 1:
            return None

        coef = self._fit_poly(all_pts)

        # Sample over the union of anchor y-ranges
        all_y = [p[1] for p in all_pts]
        y_lo, y_hi = min(all_y), max(all_y)
        if y_lo >= y_hi:
            return None

        ys = np.linspace(y_lo, y_hi, n_samples)
        xs = np.polyval(coef, ys) + shift  # horizontal translation

        synthetic = [(int(round(float(x))), int(round(float(y)))) for x, y in zip(xs, ys)]
        return synthetic

