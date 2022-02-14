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
  #cv2.imshow("src_image", src_image)
  #cv2.imshow("tw_image", tw_image)
  #cv2.waitKey()
  
  # Do an initial blur for better result
  tw_image = cv2.GaussianBlur(tw_image, (3, 3), 0)
  
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

def sb_correction(src_image, tw_image):
  # Calculate the mean and standard deviation of src_image
  src_image_hsv = cv2.cvtColor(src_image, cv2.COLOR_RGB2HSV)

  src_mean, src_std_dev = cv2.meanStdDev(src_image_hsv)

  # Calculate the mean and standard deviation of tw_image
  tw_image_hsv = cv2.cvtColor(tw_image, cv2.COLOR_RGB2HSV).astype(np.float64)

  tw_mean, tw_std_dev = cv2.meanStdDev(tw_image_hsv)

  # Do the brightness and saturation modification based on this - need to find the original paper (https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/)

  # Split the tw_image_hsv
  tw_h, tw_s, tw_v = cv2.split(tw_image_hsv)

  # Subtract difference of mean/std_dev of brightness
  tw_v -= src_mean[2] / src_std_dev[2] - tw_mean[2] / tw_std_dev[2]

  # Scale brightness by the standard deviation
  #tw_v *= src_std_dev[2] / tw_std_dev[2]
  
  # Add src image mean
  #tw_v += src_mean[2]

  # Subtract difference of mean/std_deviation of saturation
  #tw_s -= src_mean[1] / src_std_dev[1] - tw_mean[1] / tw_std_dev[1]
  tw_s -= src_mean[1] - tw_mean[1]

  # Clip the values
  tw_s = np.clip(tw_s, 0, 255)
  tw_v = np.clip(tw_v, 0, 255)


  tw_image_hsv = cv2.merge([tw_h, tw_s, tw_v])
  
  output_tw_image = cv2.cvtColor(tw_image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
  output_tw_image = cv2.cvtColor(output_tw_image, cv2.COLOR_RGB2RGBA)
  
  #Visualization
  #cv2.imshow("Source", src_image)
  #cv2.imshow("Original", tw_image)
  #cv2.imshow("Test", output_tw_image)
  #cv2.waitKey()

  # Add back the alpha channel
  output_tw_image[:, :, 3] = tw_image[:, :, 3]

  return output_tw_image

def ypr_correction(tw_image, tw_height, src_image_path, output_path):
  # Prepare the paths
  avg_ear_path = PurePath(output_path, "avg_lds.pts")
  src_landmarks_path = src_image_path.with_suffix(".pts")
  # Check if .pts annotations exists
  if not os.path.exists(avg_ear_path) or not os.path.exists(src_landmarks_path):
    return tw_image
  
  # Read in the avg_ear landmarks
  # Read the pts file
  avg_ear = np.loadtxt(str(avg_ear_path), comments = ("version:", "n_points:", "{", "}")).astype(np.float32)
  
  # Read in the src_image landmarks
  # Read the pts file
  src_landmarks = np.loadtxt(str(src_landmarks_path), comments = ("version:", "n_points:", "{", "}")).astype(np.float32)
      
  # Resize the tw_image
  new_size = int(tw_height * 80 / (tw_height * 100 / 256))
  tw_image = cv2.resize(tw_image, (new_size, new_size))

  avg_ear_image = np.zeros((265, 265, 4), dtype = np.uint8)

  # Reshape avg landmarks to the current source image size
  #for i, (landmark_x, landmark_y) in enumerate(avg_ear):
  #  landmark_x *= src_width / 256
  #  landmark_y *= src_height / 256
  #  avg_ear[i] = np.array([landmark_x, landmark_y])

  # Pad the image for compositing
  avg_ear_image = cv2.copyMakeBorder(avg_ear_image, new_size, new_size, new_size, new_size, cv2.BORDER_CONSTANT)

  # Position the tw_image to the right location
  x = int(avg_ear[44][0] - new_size / 3)
  y = int(avg_ear[44][1] - new_size / 2)
  
  # Account for the padding
  x += new_size
  y += new_size
  
  # Place the image
  avg_ear_image[y:y+new_size, x:x+new_size, :] = tw_image[:, :, :]
  
  # Crop the padding away
  avg_ear_image = avg_ear_image[new_size:-new_size, new_size:-new_size]
  
  ## Visualization avg_ear
  #for landmark in avg_ear:
  #  cv2.drawMarker(avg_ear_image, landmark.astype(np.uint64), (0, 0, 255))
  
  ## Show the image with landmarks
  #cv2.imshow("avg_ear_image", avg_ear_image)
  #cv2.waitKey()
  
  ## Visualization src_landmarks
  #show_copy = avg_ear_image.copy()
  #for landmark in src_landmarks:
  #  cv2.drawMarker(show_copy, landmark.astype(np.uint64), (255, 0, 0))
  
  ## Show the image with src_landmarks
  #cv2.imshow("avg_ear_image_src", show_copy)
  #cv2.waitKey()
  
  # Get the warp matrix
  warp_matrix = cv2.estimateAffine2D(avg_ear, src_landmarks, ransacReprojThreshold = np.Inf)[0]

  warp_tw_image = cv2.warpAffine(avg_ear_image, warp_matrix, (avg_ear_image.shape[1], avg_ear_image.shape[0]))
  
  ## Visualization src_landmarks
  #show_copy = warp_tw_image.copy()
  #for landmark in src_landmarks:
  #  cv2.drawMarker(show_copy, landmark.astype(np.uint64), (255, 0, 0))
  
  ## Show the warped image
  #cv2.imshow("Warped tw_image", show_copy)
  #cv2.waitKey()

  return warp_tw_image



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
      src_height, src_width, _ = src_image.shape
      tw_height, *_ = tw_image.shape

      ## Visualize
      #cv2.imshow("src_image", src_image)
      
      # Calculate the average sharpness and blur the tw_image to match
      tw_image = sharpness_correction(cv2.resize(src_image, (tw_height, tw_height)), tw_image)
      
      # Calculate the average birghtness and match to match
      tw_image = sb_correction(cv2.resize(src_image, (tw_height, tw_height)), tw_image)

      # Apply the affine transformation to correct for yaw, pitch and roll
      tw_image = ypr_correction(tw_image, tw_height, src_image_path, output_path)
  
      # Get the new tw_image size and resize to fit ear image
      tw_height, *_ = tw_image.shape
      new_size = int(tw_height * 100 / (tw_height * 100 / max(src_height, src_width)))
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
  parser.add_option("-t", "--tw-dir", dest = "tw_dir", help = "The path to the directory that stores the Truly Wireless earphones images.")
  parser.add_option("-r", "--random-seed", dest = "random_seed", type = "int", help = "The seed to be used in the random function, if you wish to get repeatable results.")
  (options, args) = parser.parse_args()

  if not options.input_dir or not options.annotations or not options.output_dir or not options.tw_dir:
    parser.error("The input directory (-i), annotations (-a), output directory (-o) and tw directory (-t) are required.")
  
  main(options)
