#!/usr/bin/env python
# import rospy
import math
import random
import os

def parser():

    fi = open('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/models/track/prototype_track_gan.sdf', 'r')
    fo = open('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/models/track/prototype_track_gan_mod.sdf', 'w')

    line = fi.readline()
    fo.write(line)
    cnt_left = 1
    cnt_right = 1
    cnt_orange = 1

    while line:
        line = fi.readline()
        if line.find('_left') != -1:
            place = line.find('_left') + 5
            line = line[:place] + str(cnt_left) + line[place:]
            cnt_left += 1
        if line.find('_right') != -1:
            place = line.find('_right') + 6
            line = line[:place] + str(cnt_right) + line[place:]
            cnt_right += 1
        if line.find('_big</name') != -1:
            place = line.find('_big</name') + 4
            line = line[:place] + str(cnt_orange) + line[place:]
            cnt_orange += 1

        fo.write(line)

    fi.close()
    fo.close()


if __name__ == '__main__':
    parser()
