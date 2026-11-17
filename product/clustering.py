"""纯 Python 确定性聚类工具：不依赖第三方库，结果只随输入与随机种子变化。"""

import random


def standardize(rows):
    """按列做 z-score 标准化；常数列缩为 0。返回 (scaled, means, stds)。"""
    if not rows:
        return [], [], []
    width = len(rows[0])
    means = [sum(row[i] for row in rows) / len(rows) for i in range(width)]
    stds = []
    for i in range(width):
        variance = sum((row[i] - means[i]) ** 2 for row in rows) / len(rows)
        stds.append(variance ** 0.5)
    scaled = [[(row[i] - means[i]) / stds[i] if stds[i] > 0 else 0.0
               for i in range(width)] for row in rows]
    return scaled, means, stds


def dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _kmeanspp(points, k, rng):
    centroids = [list(points[rng.randrange(len(points))])]
    while len(centroids) < k:
        weights = [min(dist2(p, c) for c in centroids) for p in points]
        total = sum(weights)
        if total <= 0:
            centroids.append(list(points[rng.randrange(len(points))]))
            continue
        pick = rng.random() * total
        acc = 0.0
        index = len(points) - 1
        for i, weight in enumerate(weights):
            acc += weight
            if pick <= acc:
                index = i
                break
        centroids.append(list(points[index]))
    return centroids


def kmeans(points, k, seed, restarts=8, max_iter=100):
    """返回 (inertia, assignments, centroids)。空簇保留上一质心，可能为空。"""
    rng = random.Random(seed)
    best = None
    for _ in range(restarts):
        centroids = _kmeanspp(points, k, rng)
        assign = [-1] * len(points)
        for _ in range(max_iter):
            new = [min(range(k), key=lambda c: (dist2(p, centroids[c]), c)) for p in points]
            if new == assign:
                break
            assign = new
            sums = [[0.0] * len(points[0]) for _ in range(k)]
            counts = [0] * k
            for a, p in zip(assign, points):
                counts[a] += 1
                for i, x in enumerate(p):
                    sums[a][i] += x
            for c in range(k):
                if counts[c]:
                    centroids[c] = [x / counts[c] for x in sums[c]]
        inertia = sum(dist2(p, centroids[a]) for p, a in zip(points, assign))
        if best is None or inertia < best[0]:
            best = (inertia, list(assign), [list(c) for c in centroids])
    return best


def silhouette(points, assign, k):
    """平均轮廓系数；样本或簇不足时返回 0。"""
    n = len(points)
    if n < 2 or k < 2:
        return 0.0
    members = {c: [i for i in range(n) if assign[i] == c] for c in range(k)}
    total = 0.0
    for i in range(n):
        own = members[assign[i]]
        a = sum(dist2(points[i], points[j]) for j in own if j != i) / max(1, len(own) - 1)
        others = [others for c, others in members.items() if c != assign[i] and others]
        b = min(sum(dist2(points[i], points[j]) for j in o) / len(o) for o in others) if others else 0.0
        m = max(a, b)
        total += (b - a) / m if m > 0 else 0.0
    return total / n


def match_clusters(old_sets, new_sets):
    """按成员重叠贪心对齐两组分群，返回 {旧序号: 新序号}，同分时按序号小者优先。"""
    candidates = sorted(
        ((len(o & n), oi, ni) for oi, o in enumerate(old_sets) for ni, n in enumerate(new_sets)),
        key=lambda t: (-t[0], t[1], t[2]))
    mapping = {}
    used_new = set()
    for size, oi, ni in candidates:
        if size == 0:
            break
        if oi in mapping or ni in used_new:
            continue
        mapping[oi] = ni
        used_new.add(ni)
    return mapping
