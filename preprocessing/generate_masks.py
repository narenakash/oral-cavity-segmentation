import os
import numpy as np
import pandas as pd
from PIL import Image
import xml.etree.ElementTree as et


data_dir = '/ssd_scratch/cvit/chocolite/PolygonSets/'
sets = [1, 5]

if not os.path.exists("/ssd_scratch/cvit/chocolite/OPMDDataset"): 
    os.makedirs("/ssd_scratch/cvit/chocolite/OPMDDataset")

# STEP ONE: GENERATE CSV

df_cols = ['img_name', 'img_height', 'img_width', 'mask_label', 'mask_points']
rows = []

for set in sets:

    file = f'Set{set}/Set{set}_MouthAnnotations.xml'

    xtree = et.parse(data_dir + file)
    xroot = xtree.getroot()

    for node in xroot:
        img_name = node.attrib.get("name")
        img_height = node.attrib.get("height") if node is not None else None
        img_width = node.attrib.get("width") if node is not None else None

        if node.find("polygon") is not None:
            mask_label = node.find("polygon").attrib.get("label") if node is not None else None
            mask_points = node.find("polygon").attrib.get("points") if node is not None else None
        else:
            mask_label = None
            mask_points = None

        rows.append({"img_name": data_dir + f'Set{set}/Originals/' + str(img_name), "img_height": img_height, "img_width": img_width, "mask_label": mask_label, "mask_points": mask_points})

out_df = pd.DataFrame(rows, columns=df_cols)

out_df = out_df.dropna()
out_df = out_df[out_df.mask_label == 'Mouth1']

# out_df.to_csv("/ssd_scratch/cvit/chocolite/OPMDDataset/opmd_dataset.csv", index=False)
out_df.to_csv("./opmd_dataset.csv", index=False)



# STEP TWO: GENERATE DATASET

def rle2mask(mask_rle, shape):
    s = mask_rle.split(',')
    starts, lengths = [np.asarray(x, dtype=float32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(shape).T


for index, row in out_df.iterrows():
    image = Image.open(row['img_name'])
    image = np.array(image)
    mask = rle2mask(row['mask_points'], (int(row['img_height']), int(row['img_width'])))
    mask = np.array(mask)

    image = Image.fromarray(image)
    image.save(f"/ssd_scratch/cvit/chocolite/OPMDDataset/{row['img_name'].split('/')[-1]}")
    mask = Image.fromarray(mask)
    mask.save(f"/ssd_scratch/cvit/chocolite/OPMDDataset/{row['img_name'].split('/')[-1].split('.')[0]}_mask.png")