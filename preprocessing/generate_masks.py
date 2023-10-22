import xml.etree.ElementTree as et
import pandas as pd


data_dir = '/ssd_scratch/cvit/chocolite/PolygonSets/'

# add files here from different sets of the dataset folder
files = ['Set1/Set1_MouthAnnotations.xml', 'Set5/Set5_MouthAnnotations.xml']

df_cols = ['img_name', 'img_height', 'img_width', 'mask_label', 'mask_points']
rows = []

for file in files:
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

        rows.append({"img_name": img_name, "img_height": img_height, "img_width": img_width, "mask_label": mask_label, "mask_points": mask_points})

out_df = pd.DataFrame(rows, columns=df_cols)

out_df = out_df.dropna()
out_df = out_df[out_df.mask_label == 'Mouth1']

out_df.to_csv("./opmd_dataset.csv", index=False)

