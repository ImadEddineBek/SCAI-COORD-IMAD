import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import ConvexHull
import alphashape
import numpy as np
import os, json, cv2, random


def convert_img_to_json(X, clustering_labels=np.array([0])):
    objs = []
    for i in range(0, clustering_labels.max() + 1):
        X_i = X
        plt.show()
        try:
            coords = np.array([np.min(X_i, axis=0), np.max(X_i, axis=0)]) * 5
            for alpha in np.arange(7.6, -5.1, -0.1):
                try:
                    hull = alphashape.alphashape(X_i, alpha)
                    hull_pts = hull.exterior.coords.xy
                    break
                except:
                    pass
            contours = []
            try:
                for i in range(len(hull_pts[0])):
                    contours.append(hull_pts[0][i] * 5)
                    contours.append(hull_pts[1][i] * 5)
            except:
                hull = ConvexHull(X_i)
                contours = []
                for simplex in hull.vertices:
                    contours.extend(X_i[simplex])
            # print(coords)
            obj = {
                "bbox": [coords[0][0], coords[0][1], coords[1][0], coords[1][1]],
                # "bbox_mode": BoxMode.XYXY_ABS,XYXY_ABS
                "segmentation": [contours],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        except:
            pass
    return objs


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(file_name):
    with open(file_name, 'rb') as f:
        x = pickle.load(f)
        # print(x)
    return x


import glob
import os
import cv2
import numpy as np
import glob
from pathlib import Path

images = glob.glob("/content/Stringed instruments (10981)/*")


def load_image(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    return img


def reduce_size_of_both_images(img_name):
    print(img_name)
    train_img = load_image(img_name)
    scale_percent = 20
    # calculate the 50 percent of original dimensions
    width = int(train_img.shape[1] * scale_percent / 100)
    height = int(train_img.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)
    output = cv2.resize(train_img, dsize)
    print("processed" + img_name)
    cv2.imwrite("processed" + img_name.replace("/content/Stringed instruments (10981)", ""), output)


def main():
    for img_name in images:
        img = load_image(img_name)
        reduce_size_of_both_images(img_name)


# main()

recrods = []
images = glob.glob("./processed_data/processed/*")[:1000]
for i, img_name in enumerate(images):
    print(img_name)
    # if i > 10:
    #     break
    img = cv2.imread(img_name)
    mask = np.zeros(img.shape[:2], np.uint8)
    h, w, _ = img.shape
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (w // 20, h // 20, w - w // 20, h - h // 20)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    X, y = np.nonzero(mask2)
    X = pd.DataFrame({'x1': y, 'x2': X}).values
    # print(len(X))
    # dbscan = DBSCAN(eps=20, min_samples=200).fit(X)
    # cluster_labels = dbscan.labels_
    # print(len(dbscan.core_sample_indices_))
    # print(len(X))
    i_ = random.sample(range(len(X)), len(X) // 10)
    record = {}
    objs = convert_img_to_json(X[i_])
    record["file_name"] = img_name.replace("./processed_data/processed/", '../Stringed Instruments (10981)')
    record["image_id"] = 0
    record["height"] = h
    record["width"] = w
    record["annotations"] = objs
    recrods.append(record)
    # break
save_object(recrods, 'recrods.pkl')


def get_all(d="train"):
    return load_object('recrods.pkl')

# from detectron2.data import DatasetCatalog, MetadataCatalog
# d = "train"
# DatasetCatalog.register("mask_" + d, lambda d=d: get_all())
# MetadataCatalog.get("mask_" + d).set(thing_classes=['figure'])
# balloon_metadata = MetadataCatalog.get("mask_train")
