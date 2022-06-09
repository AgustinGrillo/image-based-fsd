#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://github.com/rwightman/udacity-driving-reader/blob/master/script/bagdump.py

from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict
import os
import sys
import cv2
import imghdr
import argparse
import functools
import numpy as np
import pandas as pd
import rospkg

from bagutils import *


def get_outdir(base_dir, name):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def check_format(data):
    img_fmt = imghdr.what(None, h=data)
    return 'jpg' if img_fmt == 'jpeg' else img_fmt


def write_image(bridge, outdir, msg, fmt='png'):
    results = {}
    image_filename = os.path.join(outdir, str(msg.header.stamp.to_nsec()) + '.' + fmt)
    try:
        if hasattr(msg, 'format') and 'compressed' in msg.format:
            buf = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
            cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
            if cv_image.shape[2] != 3:
                print("Invalid image %s" % image_filename)
                return results
            results['height'] = cv_image.shape[0]
            results['width'] = cv_image.shape[1]
            # Avoid re-encoding if we don't have to
            if check_format(msg.data) == fmt:
                buf.tofile(image_filename)
            else:
                cv2.imwrite(image_filename, cv_image)
        else:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imwrite(image_filename, cv_image)
    except CvBridgeError as e:
        print(e)
    results['filename'] = image_filename
    return results


# sensor_msgs/Image
def camera2dict(msg, write_results, camera_dict):
    camera_dict["timestamp"].append(msg.header.stamp.to_nsec())
    camera_dict["width"].append(write_results['width'] if 'width' in write_results else msg.width)
    camera_dict['height'].append(write_results['height'] if 'height' in write_results else msg.height)
    camera_dict["frame_id"].append(msg.header.frame_id)
    camera_dict["filename"].append(write_results['filename'])


# geometry_msgs/TwistStamped
def steering2dict(msg, steering_dict):
    steering_dict["timestamp"].append(msg.header.stamp.to_nsec())
    steering_dict["angle"].append(msg.twist.angular.z)
    steering_dict["speed"].append(msg.twist.linear.x)


# ackermann_msgs/AckermannDriveStamped
def steering2dict_ack(msg, steering_dict):
    steering_dict["timestamp"].append(msg.header.stamp.to_nsec())
    steering_dict["angle"].append(msg.drive.steering_angle)
    steering_dict["speed"].append(msg.drive.speed)


def camera_select(topic, select_from):
    if topic.startswith('/l'):
        return select_from[0]
    elif topic.startswith('/c'):
        return select_from[1]
    elif topic.startswith('/r'):
        return select_from[2]
    else:
        assert False, "Unexpected topic"


def main():
    # Setup Paths
    rospack = rospkg.RosPack()
    path_imitation_learning = rospack.get_path('imitation_learning')
    output_path = path_imitation_learning + "/output"
    input_path = path_imitation_learning + "/data"

    i = 0
    while os.path.exists(output_path + "/data_%s" % i):
        i += 1

    output_folder = output_path + "/data_%s" % i

    print("Output Path:", output_folder)

    parser = argparse.ArgumentParser(description='Convert rosbag to images and csv.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default=output_folder,
                        help='Output folder')
    parser.add_argument('-i', '--indir', type=str, nargs='?', default=input_path,
                        help='Input folder where bagfiles are located')
    parser.add_argument('-f', '--img_format', type=str, nargs='?', default='jpg',
                        help='Image encode format, png or jpg')
    parser.add_argument('-m', dest='msg_only', action='store_true', help='Messages only, no images')
    parser.add_argument('-d', dest='debug', action='store_true', help='Debug print enable')
    parser.set_defaults(msg_only=False)
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    img_format = args.img_format
    base_outdir = args.outdir
    indir = args.indir
    msg_only = args.msg_only
    debug_print = args.debug

    bridge = CvBridge()

    include_images = False if msg_only else True
    include_others = False

    print("Input Path:", indir)

    print("Output Path:", base_outdir)
    if not os.path.exists(base_outdir):
        os.makedirs(base_outdir)

    filter_topics = [STEERING_TOPIC]
    if include_images:
        filter_topics += CAMERA_TOPICS
    if include_others:
        filter_topics += OTHER_TOPICS
    print(filter_topics)
    bagsets = find_bagsets(indir, filter_topics=filter_topics)
    for bs in bagsets:
        print("Processing set %s" % bs.name)
        sys.stdout.flush()

        dataset_outdir = os.path.join(base_outdir, "%s" % bs.name)
        center_outdir = get_outdir(dataset_outdir, "center")

        camera_cols = ["timestamp", "width", "height", "frame_id", "filename"]
        camera_dict = defaultdict(list)

        steering_cols = ["timestamp", "angle", "speed"]
        steering_dict = defaultdict(list)

        bs.write_infos(dataset_outdir)
        readers = bs.get_readers()
        stats_acc = defaultdict(int)

        def _process_msg(topic, msg, stats):
            timestamp = msg.header.stamp.to_nsec()

            if topic in CAMERA_TOPICS:
                outdir = center_outdir  # camera_select(topic, (left_outdir, center_outdir, right_outdir))
                if debug_print:
                    print("%s_camera %d" % (topic[1], timestamp))

                results = write_image(bridge, outdir, msg, fmt=img_format)
                results['filename'] = os.path.relpath(results['filename'], dataset_outdir)
                camera2dict(msg, results, camera_dict)
                stats['img_count'] += 1
                stats['msg_count'] += 1

            elif topic == STEERING_TOPIC:
                if debug_print:
                    print("steering %d %f" % (timestamp, msg.drive.steering_angle))

                steering2dict(msg, steering_dict)
                stats['msg_count'] += 1

        # no need to cycle through readers in any order for dumping, rip through each on in sequence
        for reader in readers:
            for result in reader.read_messages():
                _process_msg(*result, stats=stats_acc)
                if ((stats_acc['img_count'] and stats_acc['img_count'] % 1000 == 0) or
                        (stats_acc['msg_count'] and stats_acc['msg_count'] % 10000 == 0)):
                    print("%d images, %d messages processed..." %
                          (stats_acc['img_count'], stats_acc['msg_count']))
                    sys.stdout.flush()

        print("Writing done. %d images, %d messages processed." %
              (stats_acc['img_count'], stats_acc['msg_count']))
        sys.stdout.flush()

        if include_images:
            camera_csv_path = os.path.join(dataset_outdir, 'camera.csv')
            camera_df = pd.DataFrame(data=camera_dict, columns=camera_cols)
            camera_df.to_csv(camera_csv_path, index=False)

        steering_csv_path = os.path.join(dataset_outdir, 'steering.csv')
        steering_df = pd.DataFrame(data=steering_dict, columns=steering_cols)
        steering_df.to_csv(steering_csv_path, index=False)

        gen_interpolated = True
        if include_images and gen_interpolated:
            # A little pandas magic to interpolate steering/gps samples to camera frames
            camera_df['timestamp'] = pd.to_datetime(camera_df['timestamp'])
            camera_df.set_index(['timestamp'], inplace=True)
            camera_df.index.rename('index', inplace=True)
            steering_df['timestamp'] = pd.to_datetime(steering_df['timestamp'])
            steering_df.set_index(['timestamp'], inplace=True)
            steering_df.index.rename('index', inplace=True)

            merged = functools.reduce(lambda left, right: pd.merge(
                left, right, how='outer', left_index=True, right_index=True), [camera_df, steering_df])
            merged.interpolate(method='time', inplace=True)

            filtered_cols = ['timestamp', 'width', 'height', 'frame_id', 'filename',
                             'angle', 'speed']
            filtered = merged.loc[camera_df.index]  # back to only camera rows
            filtered.fillna(0.0, inplace=True)
            filtered['timestamp'] = filtered.index.astype('int')  # add back original timestamp integer col
            filtered['width'] = filtered['width'].astype('int')  # cast back to int
            filtered['height'] = filtered['height'].astype('int')  # cast back to int
            filtered = filtered[filtered_cols]  # filter and reorder columns for final output

            interpolated_csv_path = os.path.join(dataset_outdir, 'interpolated.csv')
            filtered.to_csv(interpolated_csv_path, header=True)

        print("Done")


if __name__ == '__main__':
    main()
