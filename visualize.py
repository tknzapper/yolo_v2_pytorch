import config as cfg
import utils
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def hsv2rgb(h, s=1., v=1.):
    if s == 0.0: v *= 255; return (v, v, v)
    i = int(h * 6.)  # XXX assume int() truncates!
    f = (h * 6.) - i;
    p, q, t = int(255 * (v * (1. - s))), int(255 * (v * (1. - s * f))), int(255 * (v * (1. - s * (1. - f))));
    v *= 255;
    v = int(v)
    i %= 6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)

def box_color(num_classes=cfg.num_classes):
    def _color(idx, num_classes):
        h = idx * int(360 / num_classes)
        return hsv2rgb(h / 360.)
    color = []
    for x in range(num_classes):
        color += [_color(x, num_classes)]

    return color

BOX_COLOR = box_color()

TEXT_COLOR = (255, 255, 255) # WHITE

def visualize_bbox(image, bbox, class_idx, width=3):
    # Visualize a single bounding box on the image
    xmin, ymin, xmax, ymax = utils.xywh2xxyy(bbox)

    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image

    xmin *= img.width
    ymin *= img.height
    xmax *= img.width
    ymax *= img.height

    # idx = cfg.classes.index(class_name)
    class_name = cfg.classes[class_idx]
    color = BOX_COLOR[class_idx]
    draw = ImageDraw.Draw(img)
    # Draw bbox
    draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=width)
    # Draw Label Text
    font = ImageFont.truetype(cfg.font, 12)
    text_width, text_height = font.getsize(class_name)
    draw.rectangle((xmin, ymin - int(1.3 * text_height), xmin + text_width, ymin), fill=color)
    draw.text((xmin, ymin), text=class_name, fill=TEXT_COLOR, font=font, anchor='ld')

    return img


def visualize(image, bboxes, classes):
    img = image.copy()
    for objs in range(len(bboxes)):
        bbox = bboxes[objs]
        class_idx = classes[objs]
        img = visualize_bbox(img, bbox, class_idx)
    img.show()


if __name__ == "__main__":
    print(BOX_COLOR)