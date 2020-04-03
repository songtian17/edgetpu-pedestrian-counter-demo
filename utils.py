import cv2


def get_box_color(name):
    if name == "UNKNOWN":
        bg_color = (0, 0, 255)
        txt_color = (255, 255, 255)
    else:
        bg_color = (255, 255, 255)
        txt_color = (0, 0, 0)
    return bg_color, txt_color


def draw_label(frame, rect, text):
    x_min, y_min = int(rect[0]), int(rect[1])
    bg_color, txt_color = get_box_color(text)
    label = str(text)

    # Get font size
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # Make sure not to draw label too close to top of window
    label_ymin = max(y_min, label_size[1] + 10)

    # Draw white box to put label text in
    cv2.rectangle(
        frame,
        (x_min, label_ymin - label_size[1] - 10),
        (x_min + label_size[0], label_ymin + base_line - 10),
        bg_color,
        cv2.FILLED,
    )

    # Draw label text
    cv2.putText(
        frame,
        label,
        (x_min, label_ymin - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        txt_color,
        2,
    )


def draw_centroid(frame, centroid, objectID):
    text = f"ID: {objectID}"
    cv2.putText(
        frame,
        text,
        (centroid[0] - 10, centroid[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)


def draw_detection_box(image, rect):
    (x1, y1, x2, y2) = rect
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return image


def draw_detection_boxes(image, rects):
    for rect in rects:
        image = draw_detection_box(image, rect)
    return image
