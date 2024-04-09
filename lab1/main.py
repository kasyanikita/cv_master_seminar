import cv2
import numpy as np


kernel_dilate = np.ones((5, 5), np.uint8)
kernel_erode = np.ones((3, 3), np.uint8)
colors = [(0, 0, 255), (255, 0, 0)]


def image_processing(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_img, 400, 750)
    dilated_image = cv2.dilate(canny_img, kernel_dilate, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernel_erode, iterations=1)
    
    return eroded_image


def get_lines(img):
    return cv2.HoughLinesP(img, 0.854, np.pi/190, threshold=395, minLineLength=390, maxLineGap=100)


def get_cross_point(lines):
    x1, y1, x2, y2 = lines[0][0]
    n1, m1, n2, m2 = lines[1][0]
    
    x = ((m1*n2-m2*n1)*(x2-x1) - (y1*x2-y2*x1)*(n2-n1)) / ((y2-y1)*(n2-n1) - (m2-m1)*(x2-x1))
    y = (y2-y1)/(x2-x1)*x + (y1*x2-y2*x1)/(x2-x1)
    
    return int(x), int(y)


def save_result_img(img, lines, point):
    result_img = img.copy()
    x, y = point
    
    # Draw lines
    for line, color in zip(lines, colors):
        x1, y1, x2, y2 = line[0]
        result_img = cv2.line(result_img, (x1, y1), (x2, y2), color, thickness=5)
        
    # Draw point
    result_img = cv2.circle(result_img, (x, y), radius=0, color=(0, 255, 0), thickness=10)
    
    # Save result image
    cv2.imwrite("result.png", result_img)


if __name__ == "__main__":
    original_img = cv2.imread("road1.png")
    img = image_processing(original_img)
    
    lines = get_lines(img)
    vanishing_point = get_cross_point(lines)

    save_result_img(original_img, lines, vanishing_point)
