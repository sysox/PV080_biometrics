from __future__ import annotations

import math
from collections import deque
from typing import List, Optional

import numpy as np

from .skeleton import get_8_neighbors, skel_to_bool


def crossing_number(skel01: np.ndarray, y: int, x: int) -> int:
    p = [
        skel01[y, x + 1], skel01[y - 1, x + 1], skel01[y - 1, x], skel01[y - 1, x - 1],
        skel01[y, x - 1], skel01[y + 1, x - 1], skel01[y + 1, x], skel01[y + 1, x + 1],
    ]
    s = sum(abs(int(p[i]) - int(p[(i + 1) % 8])) for i in range(8))
    return s // 2


def cluster_points(points: List[dict], dist_thresh: int = 6) -> List[dict]:
    clusters: List[List[dict]] = []
    for p in points:
        placed = False
        for c in clusters:
            cx = np.mean([q["x"] for q in c])
            cy = np.mean([q["y"] for q in c])
            if (p["x"] - cx) ** 2 + (p["y"] - cy) ** 2 <= dist_thresh ** 2 and p["type"] == c[0]["type"]:
                c.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])

    out = []
    for c in clusters:
        out.append({
            "x": int(round(np.mean([q["x"] for q in c]))),
            "y": int(round(np.mean([q["y"] for q in c]))),
            "type": c[0]["type"],
            "score": len(c),
        })
    return out


def verify_skeleton_polarity(skeleton: np.ndarray, max_white_ratio: float = 0.40) -> bool:
    white_ratio = float(np.count_nonzero(skeleton) / skeleton.size)
    return white_ratio < max_white_ratio


def compute_minutia_orientation(skeleton: np.ndarray, y: int, x: int, radius: int = 10) -> Optional[float]:
    sk = skel_to_bool(skeleton)
    h, w = sk.shape
    pts = []
    for ny in range(max(0, y - radius), min(h, y + radius + 1)):
        for nx in range(max(0, x - radius), min(w, x + radius + 1)):
            if sk[ny, nx] and not (ny == y and nx == x):
                d2 = (nx - x) ** 2 + (ny - y) ** 2
                if d2 <= radius ** 2:
                    pts.append((nx - x, ny - y))
    if len(pts) < 2:
        return None
    arr = np.asarray(pts, dtype=np.float32)
    cov = np.cov(arr.T)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]
    return float(np.arctan2(v[1], v[0]))


def _trace_branch(skel01: np.ndarray, y: int, x: int, start_y: int, start_x: int, trace_len: int = 10) -> tuple[int, int]:
    h, w = skel01.shape
    prev_y, prev_x = y, x
    cur_y, cur_x = start_y, start_x

    for _ in range(max(0, trace_len - 1)):
        nxt = []
        for ny, nx in get_8_neighbors(cur_y, cur_x, h, w):
            if skel01[ny, nx] > 0 and (ny, nx) != (prev_y, prev_x):
                nxt.append((ny, nx))
        if not nxt:
            break
        prev_y, prev_x = cur_y, cur_x
        cur_y, cur_x = nxt[0]
    return cur_y, cur_x


def compute_minutia_orientation_traced(
    skeleton: np.ndarray,
    y: int,
    x: int,
    minutia_type: str,
    trace_len: int = 10,
) -> Optional[float]:
    sk = skel_to_bool(skeleton)
    h, w = sk.shape

    neighbors = []
    for ny, nx in get_8_neighbors(y, x, h, w):
        if sk[ny, nx] > 0:
            neighbors.append((ny, nx))

    if not neighbors:
        return None

    angles = []
    for ny, nx in neighbors:
        ey, ex = _trace_branch(sk, y, x, ny, nx, trace_len=trace_len)
        angles.append(math.atan2(ey - y, ex - x))

    if minutia_type == "ending":
        return float(angles[0])

    if minutia_type == "bifurcation" and len(angles) >= 3:
        v = [(math.cos(a), math.sin(a)) for a in angles]
        d01 = v[0][0] * v[1][0] + v[0][1] * v[1][1]
        d02 = v[0][0] * v[2][0] + v[0][1] * v[2][1]
        d12 = v[1][0] * v[2][0] + v[1][1] * v[2][1]

        if d01 >= d02 and d01 >= d12:
            stem_idx = 2
        elif d02 >= d01 and d02 >= d12:
            stem_idx = 1
        else:
            stem_idx = 0

        return float((angles[stem_idx] + math.pi) % (2 * math.pi))

    return float(angles[0])


def _bfs_path_length(skel01: np.ndarray, y0: int, x0: int, y1: int, x1: int, max_steps: int = 120) -> int:
    h, w = skel01.shape
    q = deque([(y0, x0, 0)])
    seen = {(y0, x0)}

    while q:
        y, x, d = q.popleft()
        if (y, x) == (y1, x1):
            return d
        if d >= max_steps:
            continue
        for ny, nx in get_8_neighbors(y, x, h, w):
            if (ny, nx) not in seen and skel01[ny, nx] > 0:
                seen.add((ny, nx))
                q.append((ny, nx, d + 1))
    return -1


def _clean_minutiae_spurs_and_short_ridges(
    skeleton: np.ndarray,
    raw: List[dict],
    max_spur_len: int = 15,
) -> List[dict]:
    sk = skel_to_bool(skeleton)

    endings = [m for m in raw if m["type"] == "ending"]
    bifurs = [m for m in raw if m["type"] == "bifurcation"]

    e_remove = [False] * len(endings)
    b_remove = [False] * len(bifurs)

    # ending <-> bifurcation spur
    for i, e in enumerate(endings):
        if e_remove[i]:
            continue
        for j, b in enumerate(bifurs):
            if b_remove[j]:
                continue
            dist = math.hypot(e["x"] - b["x"], e["y"] - b["y"])
            if dist < max_spur_len:
                p_len = _bfs_path_length(sk, e["y"], e["x"], b["y"], b["x"], max_steps=max_spur_len)
                if 0 < p_len <= max_spur_len:
                    e_remove[i] = True
                    b_remove[j] = True
                    break

    # ending <-> ending short ridge
    max_ridge_len = max_spur_len + 5
    for i in range(len(endings)):
        if e_remove[i]:
            continue
        e1 = endings[i]
        for j in range(i + 1, len(endings)):
            if e_remove[j]:
                continue
            e2 = endings[j]
            dist = math.hypot(e1["x"] - e2["x"], e1["y"] - e2["y"])
            if dist < max_ridge_len:
                p_len = _bfs_path_length(sk, e1["y"], e1["x"], e2["y"], e2["x"], max_steps=max_ridge_len)
                if 0 < p_len <= max_ridge_len:
                    e_remove[i] = True
                    e_remove[j] = True
                    break

    out = []
    for i, e in enumerate(endings):
        if not e_remove[i]:
            out.append(e)
    for j, b in enumerate(bifurs):
        if not b_remove[j]:
            out.append(b)
    return out


def suppress_close_minutiae(points: List[dict], min_dist: int = 12) -> List[dict]:
    if not points:
        return []

    kept = []
    for p in sorted(points, key=lambda q: -float(q.get("confidence", q.get("score", 1.0)))):
        too_close = False
        for k in kept:
            if p["type"] == k["type"]:
                if math.hypot(p["x"] - k["x"], p["y"] - k["y"]) < min_dist:
                    too_close = True
                    break
        if not too_close:
            kept.append(p)
    return kept


def extract_minutiae_crossing_number(
    skeleton: np.ndarray,
    mask: np.ndarray | None = None,
    border_margin: int = 10,
    cluster_dist: int = 6,
    swap_on_inverted_polarity: bool = True,
    use_traced_orientation: bool = True,
) -> List[dict]:
    sk = skel_to_bool(skeleton)
    h, w = sk.shape
    pts = []

    polarity_ok = verify_skeleton_polarity(skeleton)

    for y in range(1 + border_margin, h - 1 - border_margin):
        for x in range(1 + border_margin, w - 1 - border_margin):
            if not sk[y, x]:
                continue
            if mask is not None and mask[y, x] == 0:
                continue

            cn = crossing_number(sk, y, x)

            if cn == 1:
                mtype = "ending"
                if swap_on_inverted_polarity and not polarity_ok:
                    mtype = "bifurcation"
                pts.append({"x": x, "y": y, "type": mtype, "confidence": 0.85})

            elif cn == 3:
                mtype = "bifurcation"
                if swap_on_inverted_polarity and not polarity_ok:
                    mtype = "ending"
                pts.append({"x": x, "y": y, "type": mtype, "confidence": 0.88})

    merged = cluster_points(pts, dist_thresh=cluster_dist)

    for m in merged:
        if use_traced_orientation:
            ang = compute_minutia_orientation_traced(skeleton, m["y"], m["x"], m["type"])
        else:
            ang = compute_minutia_orientation(skeleton, m["y"], m["x"])

        m["angle"] = ang
        m["source"] = "auto"

    return merged


def finalize_minutiae(
    raw: List[dict],
    skeleton: np.ndarray,
    max_spur_len: int = 15,
    suppress_dist: int = 12,
) -> List[dict]:
    cleaned = _clean_minutiae_spurs_and_short_ridges(
        skeleton,
        raw,
        max_spur_len=max_spur_len,
    )
    cleaned = suppress_close_minutiae(cleaned, min_dist=suppress_dist)

    for i, m in enumerate(cleaned, start=1):
        m["id"] = i
    return cleaned