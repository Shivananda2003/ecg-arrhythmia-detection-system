import cv2
import numpy as np

def remove_grid_lines(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # edge detection
    edges = cv2.Canny(gray, 50, 150)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)

    # combine grid
    grid = cv2.add(horizontal, vertical)

    # subtract grid
    cleaned = cv2.subtract(gray, grid)

    return cleaned

def extract_signal_from_image(file):

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return None

    cleaned = remove_grid_lines(img)

    
    blur = cv2.GaussianBlur(cleaned, (5, 5), 0)


    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(cleaned)

    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=1)

    height, width = mask.shape

    signal = []

    for x in range(width):

        column = mask[:, x]
        y_indices = np.where(column == 255)[0]

        if len(y_indices) > 0:
            y = np.mean(y_indices)
        else:
            y = height // 2

        signal.append(y)

    signal = np.array(signal)

    signal = cv2.GaussianBlur(signal.reshape(-1, 1), (5, 5), 0).flatten()


    signal = (signal - np.mean(signal)) / np.std(signal)

    # safety check
    if len(signal) < 50:
        return None

    return signal


def resample_to_187(signal):

    return np.interp(
        np.linspace(0, len(signal) - 1, 187),
        np.arange(len(signal)),
        signal
    )