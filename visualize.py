import config as cfg
from PIL import ImageDraw, ImageFont


def hsv2rgb(h, s=1., v=1.):
    if s == 0.0:
        v *= 255
        return (v, v, v)
    i = int(h * 6.)
    f = (h * 6.) - i
    p = int(255 * (v * (1. - s)))
    q = int(255 * (v * (1. - s * f)))
    t = int(255 * (v * (1. - s * (1. - f))))
    v = int(v * 255)
    i %= 6
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    if i == 5:
        return (v, p, q)


def box_color(num_classes=cfg.num_classes):
    def _color(idx, num_classes):
        h = idx * int(360 / num_classes)
        return hsv2rgb(h / 360.)
    color = []
    for x in range(num_classes):
        color += [_color(x, num_classes)]

    return color


BOX_COLOR = box_color()
TEXT_COLOR = (0, 0, 0) # BLACK


def visualize_bbox(image, bbox, width=3):
    # Visualize a single bounding box on the image
    xmin, ymin, xmax, ymax = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
    class_name = bbox.cls_name
    color = BOX_COLOR[bbox.cls]
    draw = ImageDraw.Draw(image)

    # Draw bbox
    draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=width)

    # Draw Label Text
    font = ImageFont.truetype(cfg.font, 12)
    text_width, text_height = font.getsize(class_name)
    draw.rectangle((xmin, ymin - int(1.3 * text_height), xmin + text_width, ymin), fill=color)
    draw.text((xmin, ymin), text=class_name, fill=TEXT_COLOR, font=font, anchor='ld')

    return image


def visualize(image, bbox):
    img = image.copy()
    for box in bbox:
        img = visualize_bbox(img, box)
    img.show()

