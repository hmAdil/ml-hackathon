"""Geometry helpers. Trips are local (tens of km), so a per-trip equirectangular
projection in metres is accurate enough; haversine is used where true distance matters."""
import numpy as np

R_EARTH = 6_371_000.0


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlat, dlon = (lat2 - lat1) / 2, (lon2 - lon1) / 2
    a = np.sin(dlat) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def make_projector(lon0: float, lat0: float):
    """Return f(lon, lat) -> (x, y) metres around (lon0, lat0)."""
    kx = np.cos(np.radians(lat0)) * R_EARTH * np.pi / 180
    ky = R_EARTH * np.pi / 180

    def proj(lon, lat):
        return (np.asarray(lon) - lon0) * kx, (np.asarray(lat) - lat0) * ky

    return proj


def point_segment_dist(px, py, ax, ay, bx, by):
    """Vectorised distance and projection fraction from points to segments A-B."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    f = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1), 0.0)
    f = np.clip(f, 0.0, 1.0)
    return np.hypot(px - (ax + f * dx), py - (ay + f * dy)), f


def bearing_deg(x1, y1, x2, y2):
    return (np.degrees(np.arctan2(x2 - x1, y2 - y1)) + 360) % 360
