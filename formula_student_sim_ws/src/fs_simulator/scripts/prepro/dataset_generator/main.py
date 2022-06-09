from texturer_v2 import generate_texture, change_texture_mode
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
# Track Model Path
path_track_model_mod = "/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/models/track/track_3_mod.sdf"
path_track_model_mod_base = "/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/models/track/track_3_mod_backup.sdf"


for iter_num in range(1, 200):
    " Noise dataset section "
    change_texture_mode('noise')
    # Randomize color and position of cones in the gazebo world
    file_track_base = open(path_track_model_mod_base, "r")
    file_track_mod = open(path_track_model_mod, "w")
    line_track = file_track_base.readline()
    while line_track:
        if line_track.strip().startswith('<uri>'):
            color = np.random.choice(['yellow', 'blue', 'orange_big'])
            line_track = '      <uri>model://cone_' + color + '</uri>\n'
        if line_track.strip().startswith('<pose>'):
            cone_pose = line_track.strip().replace('<pose>', '').split()
            cone_pose_x = float(cone_pose[0])
            cone_pose_y = float(cone_pose[1])
            # radius = np.random.uniform(0, 1.5)
            radius = np.random.triangular(0, 1.0, 1.4)
            angle = np.random.uniform(0, 2 * np.math.pi)
            delta_x = radius * np.math.cos(angle)
            delta_y = radius * np.math.sin(angle)
            new_cone_pos_x = cone_pose_x + delta_x
            new_cone_pos_y = cone_pose_y + delta_y
            line_track = '      <pose> ' + str(new_cone_pos_x) + ' ' + str(new_cone_pos_y) + ' 0 0 0 0 </pose>\n'
        file_track_mod.write(line_track)
        line_track = file_track_base.readline()
    file_track_base.close()
    file_track_mod.close()

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

    os.system('python image_gen_noise.py')

    " Simplified dataset section "
    change_texture_mode('simplified')
    os.system('python image_gen_simplified.py')

    " rgb dataset section "
    change_texture_mode('rgb')
    os.system('python image_gen_rgb.py')

    " Depth dataset section "
    os.system('python get_depth.py')

    print("Iteracion Numero: {}".format(iter_num))

