import os
import numpy as np
import pandas as pd

from ast import literal_eval
from PIL import Image, ImageDraw


if not os.path.exists("/ssd_scratch/cvit/chocolite/OPMD/"): 
    os.makedirs("/ssd_scratch/cvit/chocolite/OPMD/")

if not os.path.exists("/ssd_scratch/cvit/chocolite/OPMD/images/"): 
    os.makedirs("/ssd_scratch/cvit/chocolite/OPMD/images/")

if not os.path.exists("/ssd_scratch/cvit/chocolite/OPMD/masks/"): 
    os.makedirs("/ssd_scratch/cvit/chocolite/OPMD/masks/")

if not os.path.exists("/ssd_scratch/cvit/chocolite/OPMD/masked_images/"): 
    os.makedirs("/ssd_scratch/cvit/chocolite/OPMD/masked_images/")


df_csv = pd.read_csv("../data/opmd_dataset.csv")

# STEP TWO: GENERATE DATASET

def points2mask(points, shape):

    points = literal_eval(points)

    # round the points to the nearest integer
    points = np.round(points).astype(int)
    points = [tuple(x) for x in points]

    mask = np.zeros(shape, dtype=np.uint8)
    mask = Image.fromarray(mask)

    ImageDraw.Draw(mask).polygon(points, outline=1, fill=255)
    mask = np.array(mask)

    return mask


for index, row in df_csv.iterrows():

    print(f"Processing image {index + 1} of {len(df_csv)}")
    
    image = Image.open(row['img_dir_path'] + row['img_name'])

    mask = points2mask(row['polygon_points'], (int(row['img_height']), int(row['img_width'])))
    mask = Image.fromarray(mask)

    # resize image and mask to 512 x 512
    image = image.resize((512, 512), resample=Image.Resampling.BICUBIC)
    mask = mask.resize((512, 512), resample=Image.Resampling.BICUBIC)

    # from image and mask, generate colour masked image
    masked_image = Image.new("RGB", (512, 512))
    masked_image.paste(image, mask=mask)

    image.save(f"/ssd_scratch/cvit/chocolite/OPMD/images/{row['img_name'].split('.')[0]}.png")
    mask.save(f"/ssd_scratch/cvit/chocolite/OPMD/masks/{row['img_name'].split('.')[0]}_mask.png")
    masked_image.save(f"/ssd_scratch/cvit/chocolite/OPMD/masked_images/{row['img_name'].split('.')[0]}_masked.png")