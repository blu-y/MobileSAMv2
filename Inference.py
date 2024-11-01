import argparse
import torch
import cv2
import os
from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import sam_model_registry, SamPredictor
from typing import Any, Dict, Generator,List
import numpy as np

# check if ObjectAwareModel.tar.gz exists
# if not cat and unzip ObjectAwareModel.tar.gz.part*
if not os.path.exists('./weight/ObjectAwareModel.pt'):
    if not os.path.exists('./weight/ObjectAwareModel.tar.gz'):
        os.system("cat ./weight/ObjectAwareModel.tar.gz.part* > ./weight/ObjectAwareModel.tar.gz")
    os.system("tar -xvf ./weight/ObjectAwareModel.tar.gz -C ./weight/")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ObjectAwareModel_path", type=str, default='./PromptGuidedDecoder/ObjectAwareModel.pt', help="ObjectAwareModel path")
    parser.add_argument("--Prompt_guided_Mask_Decoder_path", type=str, default='./PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt', help="Prompt_guided_Mask_Decoder path")
    parser.add_argument("--encoder_path", type=str, default="./", help="select your own path")
    parser.add_argument("--img_path", type=str, default="./test_images/17.png", help="path to image file or directory")
    parser.add_argument("--imgsz", type=int, default=256, help="image size")
    parser.add_argument("--iou",type=float,default=0.7,help="yolo iou")
    parser.add_argument("--conf", type=float, default=0.6, help="yolo object confidence threshold")
    parser.add_argument("--retina",type=bool,default=True,help="draw segmentation masks",)
    parser.add_argument("--output_dir", type=str, default="./result/segment/", help="image save path")
    parser.add_argument("--encoder_type", type=str, default="tiny_vit", choices=['tiny_vit','sam_vit_h','mobile_sam','efficientvit_l2','efficientvit_l1','efficientvit_l0'], help="choose the model type")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    return parser.parse_args()

def create_model():
    Prompt_guided_path='./PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt'
    obj_model_path='./weight/ObjectAwareModel.pt'
    ObjAwareModel = ObjectAwareModel(obj_model_path)
    PromptGuidedDecoder=sam_model_registry['PromptGuidedDecoder'](Prompt_guided_path)
    mobilesamv2 = sam_model_registry['vit_h']()
    mobilesamv2.prompt_encoder=PromptGuidedDecoder['PromtEncoder']
    mobilesamv2.mask_decoder=PromptGuidedDecoder['MaskDecoder']
    return mobilesamv2,ObjAwareModel

def show_anns(anns, image):
    if len(anns) == 0:
        return image
    img = np.ones((anns.shape[1], anns.shape[2], 4))
    img[:,:,3] = 0
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m = m.cpu().numpy()
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    return img

def generate_image_with_annotations(image, anns):
    background = np.ones_like(image) * 255
    annotated_image = show_anns(anns, background)
    return annotated_image

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

encoder_path={'efficientvit_l2':'./weight/l2.pt',
            'tiny_vit':'./weight/mobile_sam.pt',
            'sam_vit_h':'./weight/sam_vit_h.pt',}

def main(args):
    # import pdb;pdb.set_trace()
    output_dir=args.output_dir  
    mobilesamv2, ObjAwareModel=create_model()
    image_encoder=sam_model_registry[args.encoder_type](encoder_path[args.encoder_type])
    mobilesamv2.image_encoder=image_encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mobilesamv2.to(device=device)
    mobilesamv2.eval()
    predictor = SamPredictor(mobilesamv2)
    if os.path.isdir(args.img_path):
        image_files= os.listdir(args.img_path)
    elif os.path.isfile(args.img_path):
        image_files=[os.path.basename(args.img_path)]
        args.img_path=os.path.dirname(args.img_path)+"/"
    for image_name in image_files:
        # print(image_name)
        print(args.img_path + image_name)
        image = cv2.imread(args.img_path + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        obj_results = ObjAwareModel(image,device=device,retina_masks=args.retina,imgsz=args.imgsz,conf=args.conf,iou=args.iou)
        predictor.set_image(image)
        input_boxes1 = obj_results[0].boxes.xyxy
        input_boxes = input_boxes1.cpu().numpy()
        input_boxes = predictor.transform.apply_boxes(input_boxes, predictor.original_size)
        input_boxes = torch.from_numpy(input_boxes).cuda()
        sam_mask=[]
        image_embedding=predictor.features
        image_embedding=torch.repeat_interleave(image_embedding, args.batch_size, dim=0)
        prompt_embedding=mobilesamv2.prompt_encoder.get_dense_pe()
        prompt_embedding=torch.repeat_interleave(prompt_embedding, args.batch_size, dim=0)
        for (boxes,) in batch_iterator(args.batch_size, input_boxes):
            # print("batch")
            torch.cuda.empty_cache() 
            with torch.no_grad():
                image_embedding=image_embedding[0:boxes.shape[0],:,:,:]
                prompt_embedding=prompt_embedding[0:boxes.shape[0],:,:,:]
                sparse_embeddings, dense_embeddings = mobilesamv2.prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=None,)
                low_res_masks, _ = mobilesamv2.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=prompt_embedding,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    simple_type=True,
                )
                low_res_masks=predictor.model.postprocess_masks(low_res_masks, predictor.input_size, predictor.original_size)
                # sam_mask_pre = (low_res_masks > mobilesamv2.mask_threshold)*1.0
                # print(mobilesamv2.mask_threshold)
                sam_mask_pre = (low_res_masks > 0.2)*1.0
                sam_mask.append(sam_mask_pre.squeeze(1))
        sam_mask=torch.cat(sam_mask)
        annotation = sam_mask
        areas = torch.sum(annotation, dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=True)
        show_img = annotation[sorted_indices]
        # plt.figure(figsize=(10,10))
        annotated_image = generate_image_with_annotations(image, show_img)
        # print(annotated_image.shape)
        # plt.imshow(annotated_image)
        # plt.axis('off') 
        output_name = image_name.split(".")[0] + ".png"
        output_image = annotated_image * 255
        output_image = output_image[:,:,:3].astype(np.uint8)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        cv2.imwrite(output_dir + output_name, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    args = parse_args()
    main(args)
