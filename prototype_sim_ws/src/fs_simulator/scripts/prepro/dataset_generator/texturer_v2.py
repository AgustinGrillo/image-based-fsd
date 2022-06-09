import cv2
import copy
import random
import fileinput
import sys
import rospkg

"""
Change Texture from objects
"""

rospack = rospkg.RosPack()
PATH_fs_simulator = rospack.get_path('fs_simulator')

floor_script_path = PATH_fs_simulator + "/meshes/asphalt/materials/scripts/asphalt.material"
blue_script_path = PATH_fs_simulator + "/models/cone_blue/materials/scripts/cone_blue.material"
yellow_script_path = PATH_fs_simulator + "/models/cone_yellow/materials/scripts/cone_yellow.material"
orange_script_path = PATH_fs_simulator + "/models/cone_orange_big/materials/scripts/cone_orange_big.material"

def generate_texture(object_name, path, m_texture_num, s_texture_num):

    if object_name == 'floor':
        # Change Texture
        texture_m = cv2.imread(path + "main_texture/floor_texture" + str(m_texture_num) + ".jpg")
        floor_rows, floor_cols, _ = texture_m.shape
        for _ in range(random.randint(0, 3)):
            texture_m = cv2.line(texture_m, pt1=(random.randint(0, floor_cols), random.randint(0, floor_rows)), pt2=(random.randint(0, floor_cols), random.randint(0, floor_rows)), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), thickness=random.randint(1, 20))
        for _ in range(random.randint(0, 3)):
            pt1 = (random.randint(0, floor_cols), random.randint(0, floor_rows))
            texture_m = cv2.rectangle(texture_m, pt1=pt1, pt2=(pt1[0]+random.randint(-30, 30), pt1[1]+random.randint(-30, 30)), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), thickness=-1)
        img = copy.copy(texture_m)
        # Change Texture Size
        scale_x = random.uniform(0.001, 0.01)
        scale_y = random.uniform(0.001, 0.01)
        for line in fileinput.input(floor_script_path, inplace=True):
            if line.strip().startswith('scale'):
                line = '        scale ' + str(scale_x) + ' ' + str(scale_y) + '\n'
            sys.stdout.write(line)
    else:
        mask = cv2.imread(path + "mask.png")
        texture_m = cv2.imread(path + "main_texture/m_texture" + str(m_texture_num) + ".jpg")
        texture_s = cv2.imread(path + "secondary_texture/s_texture" + str(s_texture_num) + ".jpg")

        mask_size = 2000
        red_size_m = 2000
        red_size_s = 1000

        mask = cv2.resize(mask, (mask_size, mask_size))
        texture_m = cv2.resize(texture_m, (red_size_m, red_size_m))
        texture_s = cv2.resize(texture_s, (red_size_s, red_size_s))
        new_texture_m = mask.copy()
        new_texture_s = mask.copy()

        for i in range(mask_size/red_size_m):
            for j in range(mask_size/red_size_m):
                a = i * red_size_m
                b = (i+1) * red_size_m
                c = j * red_size_m
                d = (j+1) * red_size_m
                new_texture_m[a:b, c:d, :] = texture_m
                # new_texture_w[a:b, c:d, :] = texture_w

        for i in range(mask_size/red_size_s):
            for j in range(mask_size/red_size_s):
                a = i * red_size_s
                b = (i+1) * red_size_s
                c = j * red_size_s
                d = (j+1) * red_size_s
                # new_texture_b[a:b, c:d, :] = texture_b
                new_texture_s[a:b, c:d, :] = texture_s

        black = (mask == 0)
        white = (mask == 255)
        img = mask.copy()
        img[black] = new_texture_m[black]
        img[white] = new_texture_s[white]

    if object_name == 'floor':
        cv2.imwrite(path + "texture_floor.png", img)
    elif object_name == 'blue':
        cv2.imwrite(path + "texture_b.png", img)
    elif object_name == 'yellow':
        cv2.imwrite(path + "texture_y.png", img)
    elif object_name == 'orange':
        cv2.imwrite(path + "texture_ob.png", img)


def change_texture_mode(mode):
    if mode not in ["noise", "simplified", "rgb"]:
        raise "Valid Modes: {noise, simplified, rgb}"
    if mode == "noise":
        name_prev = "texture_"
    elif mode == "simplified":
        name_prev = "original/simplified_"
    elif mode == "rgb":
        name_prev = "original/rgb_"

    for path, name in zip([floor_script_path, blue_script_path, yellow_script_path, orange_script_path], ['floor', 'b', 'y', 'ob']):
        for line in fileinput.input(path, inplace=True):
            if line.strip().startswith('texture '):
                line = '        texture ' + name_prev + name + '.png' + '\n'
            sys.stdout.write(line)
