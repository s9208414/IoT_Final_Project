import cv2
import numpy as np
import json
from PIL import Image
import io
import base64
import argparse
import os
import math 
from math import cos,sin,radians,degrees,atan2,sqrt

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
    for o,m in zip(originImgList,maskImgList):
        origin_file = os.path.join(originPath,o)
        file = os.path.join(maskPath,m)
        res = process(origin_file,file)
        resultMap[o] = res
    print('finished processing')
    return resultMap

def process(origin,mask):
    file = mask
    origin_file  = origin
    image = cv2.imread(file)
    edged = cv2.Canny(image, 10, 20)
    row_indexes, col_indexes = np.nonzero(edged) # Get coordinate of edge from pixels with nonzero value.
    tempList = [] # Store the edged pixel we have not visted.
    tabuList = [] # Store the edged pixel we have visted.
    resultList = [] # Store sorted coordinate of edged pixel.
    for i in range(0,len(row_indexes)):
        tempList.append([int(col_indexes[i]),int(row_indexes[i])])

    # Select first edged pixel to find other edged pixel which distance is the nearest to first one.
    tabuList.append(tempList[0])
    tempList.remove(tempList[0])
    while(len(tempList)>0):
        tempDis,tempCoordAndDis = findTheSmallestDis(tempList,tabuList)
        list_of_key = list(tempCoordAndDis.keys())
        list_of_value = list(tempCoordAndDis.values())
        idx = list_of_value.index(tempDis[0]) # Get the coordinate corresponding to nearest edged pixel.
        tabuList.append(json.loads(list_of_key[idx]))
        tempList.remove(json.loads(list_of_key[idx]))
    resultList = tabuList # Sorted list of coordinate.
    imageData = base64encode_img(origin_file)
    outputDict = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [{
            "label": "spine",
            "points":resultList,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }],
        "imagePath": file,
        "imageData": imageData,
        "imageHeight": image.shape[1],
        "imageWidth": image.shape[0]
    }

    return outputDict

# Output json files.
def writeFile(path,resultMap):
    print('writting..')
    for k,v in resultMap.items():
        with open(path + '/' + k + '.json', 'w') as f:
            json.dump(v, f, indent = 4)   

# Calculate Euclidean distance between the last element of "tabuList" and all elements in pixelList(tempList),create a map which key and value is coordinate and value respectively,and sort the disList ascendingly.
def findTheSmallestDis(pixelList,tabuList):
    disList = []
    coordCorrespondToDis = {}
    for i in range(0,len(pixelList)):
        dis = math.sqrt(math.pow(tabuList[-1][0] - pixelList[i][0],2) + math.pow(tabuList[-1][1] - pixelList[i][1],2))
        disList.append(dis)
        coordCorrespondToDis[str(pixelList[i])] = dis
    disList.sort()
    return disList,coordCorrespondToDis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--o',  type = str , required= True, help = 'origin image path')
    parser.add_argument('--m',  type = str , required= True, help = 'mask path')
    parser.add_argument('--j',  type = str , required= True, help = 'output path')
    args = parser.parse_args()
    originPath = args.o
    maskPath = args.m
    jsonPath = args.j
    resultMap = readFile(originPath,maskPath)
    writeFile(jsonPath,resultMap)

main()