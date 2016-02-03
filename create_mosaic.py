import csv
import cv2
import math
import numpy as np
import os
import sys
from PIL import Image

def lat_lon_to_geocentric(coords):
  """Convert Lat-Lon coordinates into geocentric (ECEF) coordinates.

  Algorithm from Wolf, Dewitt p571 Eq F-1,2,3

  Args:
   coords:  [Lon, Lat, Altitude]

  Returns:
    [[X], [Y], [Z]] (3x1 numpy matrix) in ECEF
  """
  a = 6378137  # [m]
  e2 = 0.006694380023
  dtor = math.pi / 180
  
  cos_lon = math.cos(coords[0] * dtor)
  sin_lon = math.sin(coords[0] * dtor)
  cos_lat = math.cos(coords[1] * dtor)
  sin_lat = math.sin(coords[1] * dtor)

  N = a / math.sqrt(1 - e2 * sin_lat * sin_lat);
  return np.asmatrix([[(N + coords[2]) * cos_lat * cos_lon],
                      [(N + coords[2]) * cos_lat * sin_lon],
                      [(N * (1 - e2) + coords[2]) * sin_lat]])


def geocentric_to_local(geocentric, origin):
  """Convert ECEF coordinates into local tangential coordinates

  Algoritm from Wolf and Dewitt p576 Eq F-14

  Args:
   geocentric: 3x1 numpy matrix of ECEF coordinates
   origin: [Lon Lat] of local system origin

  Returns:
    [[X], [Y], [Z]] (3x1 numpy matrix) in local tangential space
  """
  dtor = math.pi / 180
  origin_geocentric = lat_lon_to_geocentric(np.hstack((origin, [0])))

  diff = geocentric - origin_geocentric

  cos_lon = math.cos(origin[0] * dtor)
  sin_lon = math.sin(origin[0] * dtor)
  cos_lat = math.cos(origin[1] * dtor)
  sin_lat = math.sin(origin[1] * dtor)

  rotation = np.asmatrix([[          -sin_lon,            cos_lon,       0],
                          [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                          [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat]])
  return rotation * diff
  

def get_image_size(im_filename):
  """Gets the size of an image file without loading the whole image.

  Args:
    im_filename: The filename of the image
     
  Returns:
    (width, height) of the image
  """
  im = Image.open(im_filename)
  return im.size


def create_rotation_matrix(roll, pitch, yaw):
  """Creates a rotation matrix from roll, pitch and yaw angles.

  Args:
    roll: Roll angle (x-rotation) [radians]
    pitch: Pitch angle (y-rotation) [radians]
    yaw: Yaw angle (z-rotation) [radians]
  """
  rot_z = np.asmatrix([[math.cos(yaw), -math.sin(yaw), 0],
                       [math.sin(yaw),  math.cos(yaw), 0],
                       [            0,              0, 1]])
  rot_y = np.asmatrix([[ math.cos(pitch), 0, math.sin(pitch)],
                       [               0, 1,               0],
                       [-math.sin(pitch), 0, math.cos(pitch)]])
  rot_x = np.asmatrix([[1,              0,               0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll),  math.cos(roll)]])

  rot_matrix = rot_z * rot_y * rot_x
  return rot_matrix


def get_ground_corners(proj_matrix, image_width, image_height, pixel_size):
  """Get the coordinates of the projected image area on the ground.

  Args:
    proj_matrix: 3x4 camera projection matrix
    image_width: Width of the image [pixels]
    image_height: Height of the image [pixels]
    pixel_size: Side length of a pixel [m]

  Returns:
    2x4 Array of X,Y coordinates on the ground plane (z=0) [m]
  """
  half_width = 0.5 * pixel_size * image_width
  half_height = 0.5 * pixel_size * image_height
  image_corners = np.asmatrix(
      [[ -half_width,   half_width,  half_width, -half_width],
       [-half_height, -half_height, half_height, half_height],
       [          1,            1,            1,           1]])

  # We're projecting all of these images down to the ground plane (z=0), thus
  # [P00 P01 P02 P03]   [X]   [x]
  # [P10 P11 P12 P13] * [Y] = [y]
  # [P20 P21 P22 P23]   [0]   [w]
  #                     [1]
  # [P00 P01 P03]-1  [x]   [X]
  # [P10 P11 P13]  * [y] = [Y]
  # [P20 P21 P23]    [1]   [W]
  homography = np.linalg.inv(np.hstack((proj_matrix[:,0:2], proj_matrix[:,3])))
  ground_corners = homography * image_corners
  ground_corners /= ground_corners[2, :]
  return ground_corners[0:2, :]


def create_mosaic(image_fnames,
                  image_eo,
                  focal_length,
                  width,
                  height,
                  pixel_size):
  """Create a mosaic from a set of images from a single calibrated camera.

  Args:
    image_fnames: A list of image filenames
    image_eo: Exterior orientation parameters, a 6xN array of
              [X, Y, Z, Yaw, Pitch, Roll] Units of [m] and [radians]
    focal_length: Focal length of the camera [m]
    width: Image width [pixels]
    height: Image height [pixels]
    pixel_size: Size of a pixel [m]

  Returns:
    A mosaic of the input images projected to the ground. Pixel values are
    simply an average of all images that cover that pixel.
  """
  # Project images to the ground plane and get the bounding box
  min_x = float("inf")
  max_x = float("-inf")
  min_y = float("inf")
  max_y = float("-inf")
  ground_corners = []
  for eo in image_eo:
    cam_io_matrix = np.asmatrix([[focal_length,            0, 0],
                                 [           0, focal_length, 0],
                                 [           0,            0, 1]])
    rot_matrix = create_rotation_matrix(eo[5], eo[4], eo[3])
    proj_matrix = cam_io_matrix * rot_matrix * \
        np.asmatrix([[1, 0, 0, -eo[0]],
                     [0, 1, 0, -eo[1]],
                     [0, 0, 1, -eo[2]]])
    ground_corners.append(get_ground_corners(proj_matrix, width, height,
                                             pixel_size))
    min_bounds = np.amin(ground_corners[-1], axis=1)
    max_bounds = np.amax(ground_corners[-1], axis=1)
    min_x = min(min_x, min_bounds[0, 0])
    max_x = max(max_x, max_bounds[0, 0])
    min_y = min(min_y, min_bounds[1, 0])
    max_y = max(max_y, max_bounds[1, 0])

  # Put images into a local coordinate system
  ground_corners[:] = [coord - np.asmatrix([[min_x],[min_y]])
                       for coord in ground_corners]
  max_x -= min_x
  max_y -= min_y

  # Compute the average image GSD
  gsd = 0
  for eo in image_eo:
    gsd += pixel_size * eo[2] / focal_length
  gsd /= len(image_eo)

  # Create the output image
  output_cols = int(math.ceil(max_x / gsd))
  output_rows = int(math.ceil(max_y / gsd))
  output_image = np.zeros((output_rows, output_cols, 3), np.float64)
  norm_image = np.zeros(output_image.shape[0:2], np.float64)

  # Apply the image-to-ground perspective transformation to each image
  for i in range(0, len(ground_corners)):
    corners = np.int32(ground_corners[i] / gsd)
    pts1 = np.float32([[width,0], [0,0], [0, height], [width, height]])
    pts2 = np.float32(corners.transpose())
    
    im = cv2.imread(image_fnames[i], cv2.IMREAD_COLOR)
    homography = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(im, homography, (output_cols, output_rows))
    output_image += warped

    warped_ones = cv2.warpPerspective(
        np.ones(im.shape[0:2], np.float64), homography,
        (output_cols, output_rows))
    norm_image += warped_ones

  norm_image = np.maximum(norm_image, np.ones(norm_image.shape))
  output_image /= norm_image[:,:,None]

  # Draw frame outlines
  for i in range(0, len(ground_corners)):
    corners = np.int32(ground_corners[i] / gsd)
    cv2.polylines(output_image, [corners.transpose()], True, (0, 255, 0))
  cv2.imwrite("output.png", output_image)
  

if __name__ == '__main__':
  if len(sys.argv) != 2:
    sys.exit("Usage: " + sys.argv[0] + " image_eo_filename")

  image_dir = os.path.dirname(sys.argv[1])

  # Parse the input file
  image_fnames = []
  image_eo = []
  with open(sys.argv[1], 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for line in reader:
      if line[0].startswith("#"):
        continue
      if len(line) != 7:
        print "Error: Invalid format of input CSV file"
      image_fnames.append(os.path.join(image_dir, line[0]))
      image_eo.append([float(x) for x in line[1:]])

  image_eo = np.array(image_eo)
  image_eo[:, 3:] *= math.pi / 180

  origin = np.mean(image_eo[:,0:2], axis=0)
  for i in range(0, image_eo.shape[0]):
    geocentric = lat_lon_to_geocentric(image_eo[i, 0:3])
    local = geocentric_to_local(geocentric, origin)
    image_eo[i, 0:3] = local.transpose()

  focal_length = 0.02
  width, height = get_image_size(image_fnames[0])
  pixel_size = 0.035 / math.sqrt(width**2 + height**2)
  
  create_mosaic(image_fnames, image_eo, focal_length, width, height, pixel_size)
