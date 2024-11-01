import numpy as np
import cv2
import matplotlib.pyplot as plt


def apply_gray_bg(image_array, segment_mask):
    gray_image = np.ones_like(image_array) * 128
    gray_image[segment_mask] = image_array[segment_mask]
    return gray_image

def crop_image(image, segment_mask):
    # bbox = [xyxy]
    x0, y0, x1, y1 = get_bbox(segment_mask)
    cropped_image = image[y0:y1, x0:x1]
    return cropped_image

def get_bbox(segment_mask):
    return [np.min(np.where(segment_mask)[1]), np.min(np.where(segment_mask)[0]), 
            np.max(np.where(segment_mask)[1]), np.max(np.where(segment_mask)[0])]

# 이미지와 세그먼트 마스크 로드
image = cv2.imread("./test_images/3.jpg")
segment = cv2.imread("./3_annotated.png")

def get_cropped_images(image, segment):
    unique_colors = np.unique(segment.reshape(-1, segment.shape[2]), axis=0)
    unique_colors = unique_colors[~np.all(unique_colors == [255, 255, 255], axis=1)]  # 흰색 제거
    cropped_images = []
    binary_masks = []
    for color in unique_colors:
        r, g, b = color
        binary_mask = np.all(segment == [r, g, b], axis=-1)
        binary_masks.append(binary_mask)
        cropped_image = crop_image(apply_gray_bg(image, binary_mask), binary_mask)
        cropped_images.append(cropped_image)
    return cropped_images, binary_masks

# concat cropped_images
# cropped_images have different sizes
# so we need to resize them to the same size
# rest of the area will be filled with white color
def concat_cropped_images(cropped_images):
    max_height_image = max(cropped_images, key=lambda x: x.shape[0])
    max_height = max_height_image.shape[0]
    other_images = [image for image in cropped_images if image is not max_height_image]
    for image in other_images:
        h, w = image.shape[:2]
        dummy = np.ones((max_height, w, 3), dtype=np.uint8) * 255
        dummy[:h, :w] = image
        max_height_image = np.hstack([max_height_image, dummy])
    return max_height_image

cropped_images = get_cropped_images(image, segment)
concat_image = concat_cropped_images(cropped_images)
cv2.imshow("concat_image", concat_image)
cv2.waitKey(0)