import cv2
import numpy as np
from dmap import CLIP
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_model", type=str, default='ViT-B-32', choices=["ViT-B-16-SigLIP", "ViT-B-32", "ViT-B-32-256", "ViT-L-14-quickgelu"], help="clip model")
    parser.add_argument("--img_path", type=str, default="./test_images/17.png", help="path to image file")
    parser.add_argument("--seg_img_path", type=str, default="./result/segment/17.png", help="path to segment image file")
    parser.add_argument("--text_prompt", type=str, default="a fire extinguisher", help="text prompt")
    parser.add_argument("--output_dir", type=str, default="./result/sam_clip/", help="output directory")
    parser.add_argument("--mask_output_dir", type=str, default="./result/sam_clip_mask/", help="mask output directory")
    return parser.parse_args()

class SAM_CLIP:
    def __init__(self, model='ViT-B-32'):
        self.clip = CLIP(model=model)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def get_bbox(self, segment_mask):
        return [np.min(np.where(segment_mask)[1]), np.min(np.where(segment_mask)[0]), 
                np.max(np.where(segment_mask)[1]), np.max(np.where(segment_mask)[0])]

    def crop_image(self, image, segment_mask):
        # bbox = [xyxy]
        x0, y0, x1, y1 = self.get_bbox(segment_mask)
        cropped_image = image[y0:y1, x0:x1]
        return cropped_image

    def apply_gray_bg(self, image_array, segment_mask):
        gray_image = np.ones_like(image_array) * 128
        gray_image[segment_mask] = image_array[segment_mask]
        return gray_image

    def image_text_match(self, cropped_objects, text_query):
        image_features = self.clip.encode_images(cropped_objects)
        if isinstance(text_query, str): text_query = [text_query]
        text_features = self.clip.encode_text(text_query)
        probs = 100. * self.clip.similarity(image_features, text_features)
        return self.softmax(probs)
    
    def get_cropped_images(self, image, segment):
        unique_colors = np.unique(segment.reshape(-1, segment.shape[2]), axis=0)
        # remove white
        unique_colors = unique_colors[~np.all(unique_colors == [255, 255, 255], axis=1)]
        cropped_images = []
        binary_masks = []
        for color in unique_colors:
            r, g, b = color
            binary_mask = np.all(segment == [r, g, b], axis=-1)
            binary_masks.append(binary_mask)
            cropped_image = self.crop_image(self.apply_gray_bg(image, binary_mask), binary_mask)
            cropped_images.append(cropped_image)
        return cropped_images, binary_masks

    def get_id_photo_output(self, image, text, segmented_image, get_all=True):
        """
        Get the special size and background photo.

        Args:
            img(numpy:ndarray): The image array.
            size(str): The size user specified.
            bg(str): The background color user specified.
            download_size(str): The size for image saving.

        """
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cropped_images, binary_masks = self.get_cropped_images(image, segmented_image)
        scores = self.image_text_match(cropped_images, str(text))
        text_matching_masks = []
        print(scores)
        if get_all:
            for idx, score in enumerate(scores):
                if score < 0.05:
                    continue
                text_matching_mask = binary_masks[idx]
                text_matching_masks.append(text_matching_mask)
        else: # get the highest score
            max_score_idx = np.argmax(scores)
            if scores[max_score_idx] > 0.05:
                text_matching_masks.append(binary_masks[max_score_idx])
        result_image = np.copy(image)
        for text_matching_mask in text_matching_masks:
            # apply red mask with 20% transparency
            result_image[text_matching_mask] = (result_image[text_matching_mask] * 0.7 + 
                                                np.array([255, 0, 0]) * 0.3).astype(np.uint8)
        return result_image, text_matching_masks

def main(args):
    sam_clip = SAM_CLIP(model=args.clip_model)
    img = cv2.imread(args.img_path)
    seg_img = cv2.imread(args.seg_img_path)
    result_image, binary_masks = sam_clip.get_id_photo_output(img, args.text_prompt, seg_img, get_all=False)
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    if not os.path.exists(args.mask_output_dir): os.makedirs(args.mask_output_dir)
    # binary mask to RGB image
    binary_masks = np.array(binary_masks).astype(np.uint8)[0] * 255
    binary_masks = cv2.cvtColor(binary_masks, cv2.COLOR_GRAY2BGR)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    output_file = os.path.join(args.output_dir, os.path.basename(args.img_path))
    mask_output_file = os.path.join(args.mask_output_dir, os.path.basename(args.img_path))
    cv2.imwrite(output_file, result_image)
    cv2.imwrite(mask_output_file, binary_masks)

if __name__ == "__main__":
    args = parse_args()
    main(args)