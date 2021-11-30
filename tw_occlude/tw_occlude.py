#!/usr/bin/env python3
"""
Python script to occlude ear images with Truly Wireless headphones (hopefully quite accurately and real to life)
"""

import os
import shutil
import random
from optparse import OptionParser
from pathlib import PurePath
from PIL import Image

def main(options):
  # Prepare the input and output paths
  output_path = PurePath(options.output_dir)
  input_path = PurePath(options.input_dir)
  tw_path = PurePath(options.tw_dir)
  count = 0
  while os.path.exists(PurePath(output_path, input_path.name + str(count))):
    count += 1
  output_path = PurePath(output_path, input_path.name + str(count))

  # Prepare the list of TW images to later access randomly
  if options.random_seed:
    random.seed(options.random_seed)
  tw_image_paths = []
  for path, _, files in os.walk(tw_path):
    for name in files:
      tw_image_paths.append(PurePath(path, name))
  # Number of TW images
  tw_images = len(tw_image_paths)

  # Set the random seed if provided
  if options.random_seed:
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
      tw_image_path = tw_image_paths[random.randrange(0, tw_images)]

      # Load the images
      src_image = Image.open(str(src_image_path))
      tw_image = Image.open(str(tw_image_path))

      # Do the image manipulation
      # Resize the image to 40% of the src size
      _, src_height = src_image.size
      _, tw_height = tw_image.size
      new_size = int(tw_height * 70 / (tw_height * 100 / src_height))
      tw_image = tw_image.resize((new_size, new_size))
      # Place the TW image on the src
      src_image.paste(tw_image, (0, 0), tw_image)

      # Save the modified image
      src_image.save(str(src_image_path))





if __name__ == "__main__":
  # Check for input directory and output directory
  usage = "usage: %prog [options] arguments"
  parser = OptionParser()
  parser.add_option("-i", "--input-dir", dest = "input_dir", help = "The path to the ear dataset directory.")
  parser.add_option("-o", "--output-dir", dest = "output_dir", help = "The path to the directory to save the modified dataset.")
  parser.add_option("-t", "--tw-dir", dest = "tw_dir", help = "The path to the directory that stores the Truly Wirelles headphone images.")
  parser.add_option("-r", "--random-seed", dest = "random_seed", type = "int", help = "The seed to be used in the random function, if you wish to get repeatable results.")
  (options, args) = parser.parse_args()

  if not options.input_dir or not options.output_dir or not options.tw_dir:
    parser.error("The input directory (-i), output directory (-o) and tw directory (-t) are required.")
  
  main(options)
