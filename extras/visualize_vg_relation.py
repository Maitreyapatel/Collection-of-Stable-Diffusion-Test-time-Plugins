import os
import cv2
import json
import numpy as np
from PIL import Image

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

#x->w, y->h --> (x, y)
mask_image = np.zeros((h, w, 3))
objs = k['objects']
for en,obj in enumerate(objs):
    y1 = obj['y']
    y2 = y1+obj['h']
    x1 = obj['x']
    x2 = x1+obj['w']

    o = np.zeros((h, w, 3))
    o = cv2.rectangle(o, (x1, y1), (x2, y2), (255, 255, 255), 1)
    Image.fromarray(o[x1:x2, y1:y2].astype(np.uint8)).save(f"obj_{en}.png")


    mask_image = cv2.rectangle(mask_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

org_img = Image.open(id2path[region_graphs[1]['image_id']])

Image.fromarray(mask_image.astype(np.uint8)).save("mask_image.png")
org_img.save("org_image.png")