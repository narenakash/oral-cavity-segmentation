import os
import numpy as np
import pandas as pd
from skimage import draw

from PIL import Image
import xml.etree.ElementTree as et


data_dir = '/ssd_scratch/cvit/chocolite/OPMDData/'
sets = [5, 6]

# STEP ONE: GENERATE CSV

df_cols = ['img_dir_path', 'img_name', 'img_height', 'img_width', 'mask_label', 'polygon_points']
rows = []

for set in sets:

    file = f'Set{set}_20231031_onlyMouth.xml'

    xtree = et.parse(data_dir + file)
    xroot = xtree.getroot()

    for node in xroot:

        if not bool(node.attrib):
            continue

        img_name = node.attrib.get("name")
        img_height = node.attrib.get("height")
        img_width = node.attrib.get("width")

        if node.find("polygon") is not None:
            mask_label = node.find("polygon").attrib.get("label")
            polygon_points = node.find("polygon").attrib.get("points")

            # split polygon points based on semi-colon
            polygon_points = polygon_points.split(';')

            # for each element of polygon points, split based on comma
            for i in range(len(polygon_points)):
                polygon_points[i] = polygon_points[i].split(',')
                polygon_points[i] = [float(x) for x in polygon_points[i]]
            
            # convert to tuple
            polygon_points = [tuple(x) for x in polygon_points]
        else:
            mask_label = None
            polygon_points = None

        rows.append({"img_dir_path": data_dir + f'Set{set}/', "img_name": str(img_name), "img_height": img_height, "img_width": img_width, "mask_label": mask_label, "polygon_points": polygon_points})

out_df = pd.DataFrame(rows, columns=df_cols)

out_df = out_df.dropna()
out_df = out_df[out_df.mask_label == 'Mouth1']

out_df.to_csv("../data/opmd_dataset.csv", index=False)