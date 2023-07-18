import os
import cv2
import json
import numpy as np
from PIL import Image

import torch
import spacy
import en_core_web_sm
from transformers import AutoProcessor, OwlViTForObjectDetection

nlp = en_core_web_sm.load()

processor = AutoProcessor.from_pretrained("google/owlvit-base-patch16")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")

texts = [["a photo of a {}"]]


image_path = "/data/data/matt/datasets/VGENOME/VG_100K"
id2path = {}
for fname in os.listdir(image_path):
    id2path[int(fname.split('.')[0])] = os.path.join(image_path, fname)


with open("/data/data/matt/datasets/VGENOME/region_graphs.json", "r") as h:
    region_graphs = json.load(h)

with open("/data/data/matt/datasets/VGENOME/image_data.json", "r") as h:
    image_data = json.load(h)

size = {}
for d in image_data:
    size[d['image_id']] = (d['height'], d['width'])

h,w = size[region_graphs[1]['image_id']]
k = region_graphs[1]['regions'][1]

if len(k['relationships'])>0:
    print("Caption: ", k['phrase'])

bbox_phrases = [w for w in nlp(k['phrase']) if w.pos_ == 'NOUN']
print(bbox_phrases)

image = Image.open(id2path[region_graphs[1]['image_id']])

#x->w, y->h --> (x, y)
mask_image = np.zeros((h, w, 3))
region_image = np.zeros((h, w, 3))

y1 = k['y']
y2 = y1+k['height']
x1 = k['x']
x2 = x1+k['width']
region_image[y1:y2, x1:x2] = 255
Image.fromarray(region_image.astype(np.uint8)).save("region_image.png")

for en,phrase in enumerate(bbox_phrases): 
    texts = [[f"a photo of a {phrase}"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs=outputs, threshold=0.08, target_sizes=target_sizes
    )

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")


        x1 = int(box[0])
        x2 = int(box[2])
        y1 = int(box[1])
        y2 = int(box[3])

        o = np.zeros((h, w, 3))
        o = cv2.rectangle(o, (x1, y1), (x2, y2), (255, 255, 255), 1)
        Image.fromarray(o[x1:x2, y1:y2].astype(np.uint8)).save(f"obj_{en}.png")

        mask_image[y1:y2, x1:x2] = 255
    # mask_image = cv2.rectangle(mask_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

org_img = Image.open(id2path[region_graphs[1]['image_id']])

Image.fromarray(mask_image.astype(np.uint8)).save("mask_image.png")
org_img.save("org_image.png")