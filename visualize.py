import json
import torch
from PIL import Image, ImageDraw, ImageFont

def draw_boxes_and_labels(image_path, boxes, gt, output_path="output.jpg"):
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size
        try:
          font = ImageFont.truetype("arial.ttf", 30)
        except OSError:
          font = ImageFont.load_default(30)
        
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax, score, _ = box
            cx, cy = (xmin+xmax)/2, (ymin+ymax)/2     
            label = str(round(score.item(), 2))
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill="red")
            # draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 0, 0, 10), width=1)
            # draw.text((cx, cy - 3), label, fill="white", font=font)
        draw.text((100, 10), str(boxes.shape[0]), fill="red", font=font)

        for i, box in enumerate(gt):
            cx, cy, width, height, _ = box 
            cx, cy = cx*img_width, cy*img_height
            xmin, ymin, xmax, ymax = cx - width*img_width/2, cy - height*img_height/2, cx + width*img_width/2, cy + height*img_height/2
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill="green")
            # draw.rectangle((xmin, ymin, xmax, ymax), outline=(0, 255, 0, 10), width=1)
        draw.text((100, 40), str(len(gt)), fill="green", font=font)

        image.save(output_path)

    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def iou_postprocess(bbox):
    idx = []
    retain = []
    cx, cy = (bbox[:, 0] + bbox[:, 2])/2, (bbox[:, 1] + bbox[:, 3])/2
    center_points = torch.stack((cx, cy), axis=1)
    axis_ind = 1 - int((max(center_points[:, 0]) - min(center_points[:, 0])) > (max(center_points[:, 1]) - min(center_points[:, 1])))
    bbox = bbox[torch.sort(center_points[:, axis_ind], dim=0)[1]]
    i = 0
    j = 1
    while(j<bbox.shape[0]):
        x1_min, y1_min, x1_max, y1_max = bbox[i, :4]
        x2_min, y2_min, x2_max, y2_max = bbox[j, :4]
        cx1, cy1 = (x1_min+x1_max)/2, (y1_min+y1_max)/2 
        cx2, cy2 = (x2_min+x2_max)/2, (y2_min+y2_max)/2 
        
        if axis_ind == 0:
            range_xmin = cx1 - (x1_max - x1_min)/4
            range_xmax = cx1 + (x1_max - x1_min)/4
            between = cx2 > range_xmin and cx2 < range_xmax

        else:
            range_ymin = cy1 - (y1_max - y1_min)/4
            range_ymax = cy1 + (y1_max - y1_min)/4
            between = cy2 > range_ymin and cy2 < range_ymax

        if between:
            if bbox[i, 4] < bbox[j, 4]:
                idx.append(i)
                i = j
            else:
                idx.append(j)
        else:
            i = i + 1
        
        j = j + 1

    for i in range(bbox.shape[0]):
        if i not in idx:
            retain.append(bbox[i])
                
    chosen = torch.stack([r for r in retain], dim=0)
    return chosen

test_gt = {
    "dense99_png.rf.867886e0fe37a21f7cfb645e5cb1d40d.jpg" : 24,
    "lab00001_jpg.rf.7817d0d3cdcb8287e01d899197034642.jpg" : 3,
    "IMG_4904_JPG.rf.c53e8612b38daa7b6a83457c9208a59b.jpg" : 43
} 

if __name__ == "__main__":
    base_path = "./results"
    img_paths = "./surgcount-hd/test/"
    annotations = "./data/surgcount-hd/handle_coco/surgcount_hd_test_coco.json"

    pkl = torch.load(f"{base_path}/results-0.pkl")
    gt_info = pkl['gt_info']
    res_info = pkl['res_info']

    sigma = 0.26

    root = f"./{base_path}/output/"
    img_names = []

    with open(annotations, 'r') as file:
            data = json.load(file) 
            img_names = [c['file_name'] for c in data['images']]

    diff = 0
    idx = 0
    for img_path, box, gtbox in zip(img_names, res_info, gt_info):
        mask = box[:, 4] >= sigma
        chosen = box[mask]
        # chosen = iou_postprocess(chosen)
        
        fname = img_path.split("/")[-1]
        draw_boxes_and_labels(img_paths+img_path, chosen, gtbox, root + fname)
        diff = diff + abs(chosen.shape[0] - test_gt[img_path])
        idx += 1

    print(f"MAE ({idx} images): {diff/idx}")




