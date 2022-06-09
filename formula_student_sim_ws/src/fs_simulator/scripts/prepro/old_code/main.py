from texturer_v1 import change_texture
from texturer_v2 import generate_texture
import os
import random
import time
import numpy as np
import fileinput
import sys


# Set random seed based on clock time
random.seed(time.time())
np.random.seed(int(time.time()))

# Set random limits
random_floor = 758
random_blue = 94
random_yellow = 78
random_orange = 77
random_white = 88
random_black = 65

# Set model paths
path_floor = "/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/meshes/asphalt/materials/textures/"
path_blue = "/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/models/cone_blue/materials/textures/"
path_yellow = "/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/models/cone_yellow/materials/textures/"
path_orange = "/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/models/cone_orange_big/materials/textures/"
# Gazebo world path
path_world = "/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/worlds/track_def.world"
# URDF path
path_urdf = "/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/urdf/fs_prepro.urdf.xacro"


for iter_num in range(1, 350):
    # Set random textures
    generate_texture('floor', path_floor, random.randint(1, random_floor), None)
    generate_texture('blue', path_blue, random.randint(1, random_blue), random.randint(1, random_white))
    generate_texture('yellow', path_yellow, random.randint(1, random_yellow), random.randint(1, random_black))
    generate_texture('orange', path_orange, random.randint(1, random_orange), random.randint(1, random_white))

    # Modify ambient light color and intensity
    mean_ambient = random.randint(120, 255)
    RGB_ambient = np.random.normal(mean_ambient, 10, 3).clip(0, 255).astype(int)
    # Modify ambient light direction
    pose_ambient = np.random.uniform(-1.1, 1.1, 3)
    # Modify gaussian noise stdv
    noise_stdv = np.random.uniform(0, 0.02, 1)

    # Replace values in file
    for line in fileinput.input(path_world, inplace=True):
        if line.strip().startswith('<pose>0 0 10 '):
            line = '      <pose>0 0 10 ' + str(pose_ambient[0]) + ' ' + str(pose_ambient[1]) + ' ' + str(pose_ambient[2]) + '</pose>\n'
        if line.strip().startswith('<ambient>'):
            line = '      <ambient>' + str(RGB_ambient[0]) + ' ' + str(RGB_ambient[1]) + ' ' + str(RGB_ambient[2]) + ' 255</ambient>\n'
        sys.stdout.write(line)

    for line in fileinput.input(path_urdf, inplace=True):
        if line.strip().startswith('<stddev>'):
            line = '          <stddev>' + str(noise_stdv[0]) + '</stddev>\n'
        sys.stdout.write(line)


    os.system('python image_gen_v4.py')
    print("Iteracion Numero: {}".format(iter_num))
