import numpy as np
import cv2


_colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255)
}


def draw_bbox(img, bbox, color='color'):
    if isinstance(color, str):
        color = _colors[color]

    x1, y1, x2, y2 = bbox
    pt1 = (int(x1), int(y1))
    pt2 = (int(x2), int(y2))
    img = cv2.rectangle(img.copy(), pt1, pt2, color, 2)
    return img
