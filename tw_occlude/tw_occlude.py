#!/usr/bin/env python3
"""
Python script to occlude ear images with Truly Wireless headphones (hopefully quite accurately and real to life)
"""

import os
import shutil
import random
import json
import cv2
import numpy as np
from optparse import OptionParser
from pathlib import PurePath

def side(annotations, src_image_path):
  if annotations == "L":
    # Access the cropped image for the left ear
    return 0
  elif annotations == "R":
    # Access the cropped image for the left ear
    return 1
  elif annotations:
    NotImplementedError()


def sharpness_correction(src_image, tw_image):
  # Calculate the mean sharpness in the source image
  src_image_gray = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
  src_laplacian = cv2.Laplacian(src_image_gray, cv2.CV_64F)
  _, sigma = cv2.meanStdDev(src_laplacian)

  sharpness_value = sigma[0][0] ** 2
  #print("Original src:", sharpness_value)

  # Blur untill the tw_image is around the same or lower sharpness
  tw_image_gray = cv2.cvtColor(tw_image, cv2.COLOR_RGB2GRAY)
  tw_laplacian = cv2.Laplacian(tw_image_gray, cv2.CV_64F)
  _, sigma = cv2.meanStdDev(tw_laplacian)

  tw_corrected_sharpness = sigma[0][0] ** 2
  #print("Original tw:", tw_corrected_sharpness)
  #    
  ## Show the image
  #cv2.imshow("tw_image", tw_image)
  #cv2.waitKey()
  
  while tw_corrected_sharpness > sharpness_value / 2:
    #tw_image = cv2.blur(tw_image, (3, 3))
    tw_image = cv2.GaussianBlur(tw_image, (3, 3), 0)

    tw_image_gray = cv2.cvtColor(tw_image, cv2.COLOR_RGB2GRAY)
    tw_laplacian = cv2.Laplacian(tw_image_gray, cv2.CV_64F)
    _, sigma = cv2.meanStdDev(tw_laplacian)

    tw_corrected_sharpness = sigma[0][0] ** 2
    #print("Correction step:", tw_corrected_sharpness)

    ## Show the image
    #cv2.imshow("tw_image blurred", tw_image)
    #cv2.waitKey()

  return tw_image



def main(options):
  # Prepare the input and output paths
  output_path = PurePath(options.output_dir)
  input_path = PurePath(options.input_dir)
  tw_path = PurePath(options.tw_dir)
  count = 0
  while os.path.exists(PurePath(output_path, input_path.name + str(count))):
    count += 1
  output_path = PurePath(output_path, input_path.name + str(count))

  """
  Old Code
  # Prepare the list of TW images to later access randomly
  if options.random_seed:
    random.seed(options.random_seed)
  tw_image_paths = []
  for path, _, files in os.walk(tw_path):
    for name in files:
      tw_image_paths.append(PurePath(path, name))
  # Number of TW images
  tw_images = len(tw_image_paths)
  """

  # Read the tw images definitions
  tw_file = open(PurePath(tw_path, "definitions.json"))
  tw_definitions = json.load(tw_file)
  tw_image_paths = [(PurePath(tw_path, tw_definition["cropped_path_left"]), PurePath(tw_path, tw_definition["cropped_path_right"])) for tw_definition in tw_definitions["images"]]
  # Number of TW images
  tw_images = len(tw_image_paths)

  # Set the random seed if provided
  if options.random_seed is not None:
    random.seed(options.random_seed)

  # Copy the entire dataset to the output_dir and continue operations there
  shutil.copytree(input_path, output_path)

  # Get all the files of the input dataset and apply a random TW image
  for path, _, files in os.walk(output_path):
    for name in files:
      # Prepare the paths
      src_image_path = PurePath(path, name)
      # Check if the file is an image
      if src_image_path.suffix not in [".png", ".jpg", ".jpeg"]:
        continue
      tw_image_path = tw_image_paths[random.randrange(0, tw_images)][side(options.annotations, src_image_path)]

      # Load the images
      src_image = cv2.imread(str(src_image_path), cv2.IMREAD_UNCHANGED)
      tw_image = cv2.imread(str(tw_image_path), cv2.IMREAD_UNCHANGED)

      # Do the image manipulation
      # Resize the image to 40% of the src size
      src_height, src_width, _ = src_image.shape
      tw_height, *_ = tw_image.shape
      
      # Calculate the average sharpness and blur the tw_image to match
      tw_image = sharpness_correction(cv2.resize(src_image, (tw_height, tw_height)), tw_image)

      # Resize the blurred image
      new_size = int(tw_height * 80 / (tw_height * 100 / src_height))
      tw_image = cv2.resize(tw_image, (new_size, new_size))

      # Pad the image for compositing
      src_image = cv2.copyMakeBorder(src_image, new_size, new_size, new_size, new_size, cv2.BORDER_CONSTANT)

      # Calculate the optimal placement, x is inverted for left to right ear
      x = int(src_width / 3 - new_size / 2)
      if (options.annotations == "R"):
        x = int(1.8 * src_width / 3 - new_size / 2)
      y = int(src_height * 1.7 / 3 - new_size / 2)
      
      # Account for the padding
      x += new_size
      y += new_size

      # Calculate the alpha channel
      alpha_tw_image = tw_image[:, :, 3] / 255

      # Place the TW image on the src
      for colour in range(3):
        # Mask
        src_image[y:y+new_size, x:x+new_size, colour] = alpha_tw_image * tw_image[:, :, colour] + (1 - alpha_tw_image) * src_image[y:y+new_size, x:x+new_size, colour]

      # Crop the padding away
      src_image = src_image[new_size:-new_size, new_size:-new_size]

      ## Show the image
      #cv2.imshow("Output", src_image)
      #cv2.waitKey()

      # Save the modified image
      cv2.imwrite(str(src_image_path), src_image)



if __name__ == "__main__":
  # Check for input directory and output directory
  usage = "usage: %prog [options] arguments"
  parser = OptionParser()
  parser.add_option("-i", "--input-dir", dest = "input_dir", help = "The path to the ear dataset directory.")
  parser.add_option("-a", "--annotations", dest = "annotations", help = "The mode of handling left and right ears in the ear dataset (L, R, <annotation_filename>).")
  parser.add_option("-o", "--output-dir", dest = "output_dir", help = "The path to the directory to save the modified dataset.")
  parser.add_option("-t", "--tw-dir", dest = "tw_dir", help = "The path to the directory that stores the Truly Wirelles headphone images.")
  parser.add_option("-r", "--random-seed", dest = "random_seed", type = "int", help = "The seed to be used in the random function, if you wish to get repeatable results.")
  (options, args) = parser.parse_args()

  if not options.input_dir or not options.annotations or not options.output_dir or not options.tw_dir:
    parser.error("The input directory (-i), annotations (-a), output directory (-o) and tw directory (-t) are required.")
  
  main(options)
