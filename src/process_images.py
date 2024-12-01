#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os

# image = np.asarray(bytearray(resp.read()), dtype="uint8")
# image = cv2.imdecode(image, cv2.IMREAD_COLOR)
# image = cv2.resize(image,(224,224))

new_path = "/Users/marcus/Desktop/575/project/EEG-ImageNet-Dataset/data/imageNet_images/"
old_path = "/Users/marcus/Desktop/575/project/EEG-ImageNet-Dataset/data/imageNet_images_old/"
output_size = 224


def convert_image(path):
    old_image_file = os.path.join(old_path, path.split('_')[0], path)
    new_image_file = os.path.join(new_path, path.split('_')[0], path)

    new_synset_path = os.path.join(new_path, path.split('_')[0])
    if not os.path.exists(new_synset_path):
        os.makedirs(new_synset_path)

    im = cv2.imread(old_image_file, cv2.IMREAD_COLOR)
    if im is None:
        print(f"Error: Could not read image {old_image_file}")
        return False

    try:
        im_resize = cv2.resize(im, (output_size, output_size))
        cv2.imwrite(new_image_file, im_resize)
        print(f"Converted and saved: {new_image_file}")
        return True
    except Exception as e:
        print(f"Error processing {old_image_file}: {e}")






# def main():       
#     startp = 0

#     # read in the wordnet id list
#     infile = open("/Users/marcus/Desktop/575/project/EEG-ImageNet-Dataset/data/imageNet_images/wnids1000.txt","r")
#     lines = infile.readlines()
#     wnids = []
#     for line in lines:
#         wnids.append(line.strip('\n').split(' ')[0])
#     wnids = wnids[startp:]
#     # download images


#     for i in range(len(wnids)):
#         print("wnids %s"%(wnids[i]))
#         # url = "https://image-net.org/data/winter21_whole/" + wnids[i] + ".tar"


#         # extract_to = "/Users/marcus/Desktop/575/project/EEG-ImageNet-Dataset/data/imageNet_images/" + wnids[i] + "/"


#         if extract_to and not os.path.exists(extract_to):
#             os.makedirs(extract_to)
            