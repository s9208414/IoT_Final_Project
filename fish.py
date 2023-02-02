import cv2
import numpy as np
import json
from PIL import Image
import io
import base64
import argparse
import os

# Convert an image to Base64.
def base64encode_img(image_path):
    src_image = Image.open(image_path)
    output_buffer = io.BytesIO()
    src_image.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str

# Read origin image and mask.
def readFile(originPath,maskPath):
    originImgList = os.listdir(originPath)
    originImgList.sort(key=lambda x:int(x.split(' ')[7][:-4]))
    maskImgList = os.listdir(maskPath)
    maskImgList.sort(key=lambda x:int(x.split(' ')[7][:-8]))
    resultMap = {}
    imgMap = {}
    for o,m in zip(originImgList,maskImgList):
        origin_file = os.path.join(originPath,o)
        file = os.path.join(maskPath,m)
        res,img = process(origin_file,file)
        resultMap[o] = res
        imgMap[o] = img
    print('finished processing')
    return resultMap,imgMap

def process(origin,mask):
    file = mask
    origin_file  = origin
    image = cv2.imread(file)
    origin_img = cv2.imread(origin_file)
    edged = cv2.Canny(image, 10, 20)
    row_indexes, col_indexes = np.nonzero(edged) # Get coordinate of edge from pixels with nonzero value 
    tempList = []
    for i in range(0,len(row_indexes)):
        tempList.append([int(col_indexes[i]),int(row_indexes[i])])
    imageData = base64encode_img(origin_file)
    maskData = base64encode_img(file)
    outputDict = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [{
            "label": "spine",
            "points":tempList,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }],
        "imagePath": file,
        "imageData": maskData,
        "imageHeight": image.shape[1],
        "imageWidth": image.shape[0]
    }
    pts = np.array(tempList)
    pts = pts.reshape((-1, 1, 2))
    img = cv2.polylines(image, [pts], False, (0,255,255))
    return outputDict,img

# Output json and png files.
def writeFile(path,resultMap,imgMap):
    print('writting..')
    for k,v in resultMap.items():
        with open(path + '/' + k + '.json', 'w') as f:
            json.dump(v, f, indent = 4)   
    for k,v in imgMap.items():
        cv2.imwrite(path + '/' + k + '.png',v) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--o',  type = str , required= True, help = 'origin image path')
    parser.add_argument('--m',  type = str , required= True, help = 'mask path')
    parser.add_argument('--j',  type = str , required= True, help = 'output path')
    args = parser.parse_args()
    originPath = args.o
    maskPath = args.m
    jsonPath = args.j
    resultMap,imgMap = readFile(originPath,maskPath)
    writeFile(jsonPath,resultMap,imgMap)

main()
