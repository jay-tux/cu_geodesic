#!/usr/bin/python3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas


def overlay_segm(segm_file):
    with open(segm_file) as f:
        line1 = f.readline()
        l1spl = line1.split(' ')
        start = (float(l1spl[0]), float(l1spl[1]))

        def helper(pt):
            tmp = pt.split(' ')
            return float(tmp[0]), float(tmp[1])

        line2 = f.readline()
        hull_pts = [helper(p) for p in line2.split(';')]
        hull_x = [pt[0] for pt in hull_pts]
        hull_x.append(hull_x[0])
        hull_y = [pt[1] for pt in hull_pts]
        hull_y.append(hull_y[0])

        holes = []
        for line in f.readlines():
            hole_pts = [helper(p) for p in line.split(';')]
            hole_x = [pt[0] for pt in hole_pts]
            hole_x.append(hole_x[0])
            hole_y = [pt[1] for pt in hole_pts]
            hole_y.append(hole_y[0])
            holes.append((hole_x, hole_y))

    plt.plot(hull_x, hull_y)
    for (hole_x, hole_y) in holes:
        plt.plot(hole_x, hole_y)
    plt.plot([start[0]], [start[1]], 'o', color='y')


def color_for(dist, _min, _max):
    symbols = '0123456789ABCDEF'
    if dist == -1.0:
        return '#000000'
    if dist < _min:
        dist = _min
    if dist > _max:
        dist = _max
    factor = (dist - _min) / (_max - _min)
    red = factor * 255
    blue = (1 - factor) * 255
    if red >= 256 or blue >= 256:
        print(f'Idx: red={red}, blue={blue}')
    r1, r2, b1, b2 = [int(x) for x in [red / 16, red % 16, blue / 16, blue % 16]]
    return f'#{symbols[r1]}{symbols[r2]}00{symbols[b1]}{symbols[b2]}'


def draw_points(ref_file):
    filename = ref_file + ".csv"
    frame = pandas.read_csv(filename)
    _points = [
        (float(frame['point x'][point]),
         float(frame['point y'][point]),
         float(frame['distance'][point]))
        for point in frame.index
    ]
    points = [
        [p[0] for p in _points],
        [p[1] for p in _points],
        [p[2] for p in _points]
    ]
    min_d = min(points[2])
    max_d = max(points[2])

    idx = np.argmax(points[2])
    tmp = [color_for(p, min_d, max_d) for p in points[2]]
    points[2] = tmp
    plt.scatter(points[0], points[1], c=points[2], marker='*', linewidths=0.5)
    plt.scatter([points[0][idx]], [points[1][idx]], c='y', marker='o')


draw_points(sys.argv[1])
overlay_segm(sys.argv[1])
plt.show()
