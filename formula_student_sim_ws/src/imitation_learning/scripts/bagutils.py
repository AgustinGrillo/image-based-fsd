# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

from __future__ import print_function
from six import iteritems
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict
import os
import sys
import fnmatch
import subprocess
import cv2
import yaml
import rosbag
import datetime


SEC_PER_NANOSEC = 1e9
MIN_PER_NANOSEC = 6e10

LEFT_CAMERA_TOPIC = "/left_camera/image_color"
# CENTER_CAMERA_TOPIC = "/cam/camera_/image_raw"
#CENTER_CAMERA_TOPIC = "/img_node/intensity_image"
#CENTER_CAMERA_TOPIC = "/openmv_cam/image/raw"

CENTER_CAMERA_TOPIC="/fs/c1/image_raw"

RIGHT_CAMERA_TOPIC = "/right_camera/image_color"
LEFT_CAMERA_COMPRESSED_TOPIC = LEFT_CAMERA_TOPIC + "/compressed"
CENTER_CAMERA_COMPRESSED_TOPIC = CENTER_CAMERA_TOPIC + "/compressed"
RIGHT_CAMERA_COMPRESSED_TOPIC = RIGHT_CAMERA_TOPIC + "/compressed"
CAMERA_TOPICS = [CENTER_CAMERA_TOPIC]
CENTER_CAMERA_TOPICS = [CENTER_CAMERA_TOPIC, CENTER_CAMERA_COMPRESSED_TOPIC]
# STEERING_TOPIC = "/cmd_vel_stamped"
# STEERING_TOPIC = "/joy_teleop/cmd_vel_stamped"
#STEERING_TOPIC = "/racecar/ackermann_cmd_mux/output"

STEERING_TOPIC ="fs/cmd_vel_stamped"

GPS_FIX_TOPIC = "/vehicle/gps/fix"
GPS_FIX_NEW_TOPIC = "/fix"

WHEEL_SPEED_TOPIC = "/vehicle/wheel_speed_report"
THROTTLE_TOPIC = "/vehicle/throttle_report"
BRAKE_TOPIC = "/vehicle/brake_report"
GEAR_TOPIC = "/vehicle/gear_report"
IMU_TOPIC = "/vehicle/imu/data_raw"

OTHER_TOPICS = [
    WHEEL_SPEED_TOPIC, THROTTLE_TOPIC, BRAKE_TOPIC, GEAR_TOPIC, IMU_TOPIC]

CAMERA_REMAP_LCCL = {
    LEFT_CAMERA_TOPIC: CENTER_CAMERA_TOPIC,
    LEFT_CAMERA_COMPRESSED_TOPIC: CENTER_CAMERA_COMPRESSED_TOPIC,
    CENTER_CAMERA_TOPIC: LEFT_CAMERA_TOPIC,
    CENTER_CAMERA_COMPRESSED_TOPIC: LEFT_CAMERA_COMPRESSED_TOPIC,
    'left_camera': 'center_camera',
    'center_camera': 'left_camera',
}


def check_remap_hack(filename):
    if fnmatch.fnmatch(filename, "2016-10-25*.bag"):
        print(filename, 'matches remap hack.')
        return CAMERA_REMAP_LCCL
    else:
        return {}


def get_bag_info(bag_file, nanosec=True):
    info = yaml.load(subprocess.Popen(
        ['rosbag', 'info', '--yaml', bag_file],
        stdout=subprocess.PIPE).communicate()[0])
    if nanosec:
        if 'start' in info:
            info['start'] = int(info['start']*1e9)
        if 'end' in info:
            info['end'] = int(info['end']*1e9)
        if 'duration' in info:
            info['duration'] = int(info['duration']*1e9)
    return info


def get_topic_names(bag_info_yaml):
    topic_names = []
    topics = bag_info_yaml['topics']
    for t in topics:
        topic_names.append(t['topic'])
    return topic_names


def ns_to_str(timestamp_ns):
    secs = timestamp_ns / 1e9
    dt = datetime.datetime.fromtimestamp(secs)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')


class BagReader(object):
    def __init__(self, bagfiles, topics, remap_camera={}):
        self.bagfiles = bagfiles
        self.topics = topics
        self._remap_camera = remap_camera

    def read_messages(self):
        for f in self.bagfiles:
            with rosbag.Bag(f, "r") as bag:
                for topic, msg, _ in bag.read_messages(topics=self.topics):
                    if self._remap_camera and topic in self._remap_camera:
                        topic = self._remap_camera[topic]
                        msg.header.frame_id = self._remap_camera[msg.header.frame_id]
                    yield topic, msg


JOIN_THRESH_NS = 10 * MIN_PER_NANOSEC


class BagSet(object):

    def __init__(self, name, bagfiles, filter_topics, remap_camera={}):
        self.name = name
        self.files = sorted(bagfiles)
        self.infos = []
        self.topic_map = defaultdict(list)
        self.start_time = None
        self.end_time = None
        self._remap_camera = remap_camera
        self._process_infos(filter_topics)

    def _process_infos(self, filter_topics):
        for f in self.files:
            print("Extracting bag info %s" % f)
            sys.stdout.flush()
            info = get_bag_info(f)
            if 'start' not in info or 'end' not in info:
                print('Ignoring info %s without start/end time' % info['path'])
                continue
            if self._remap_camera and check_remap_hack(os.path.basename(f)):
                info['remap'] = self._remap_camera
            info_start = info['start']
            info_end = info['end']
            if not self.start_time or not self.end_time:
                self._extend_range(info_start, info_end)
            elif (info_start - JOIN_THRESH_NS) <= self.end_time and self.start_time <= (info_end + JOIN_THRESH_NS):
                self._extend_range(info_start, info_end)
            else:
                print('Orphaned bag info time range, are there multiple datasets in same folder?')
                continue
            self.infos.append(info)
            if self._remap_camera:
                filter_topics = self._filter_topics_remap(filter_topics)
            filtered = [x['topic'] for x in info['topics'] if not filter_topics or x['topic'] in filter_topics]
            gps_fix_replace = False
            if GPS_FIX_NEW_TOPIC in filtered and GPS_FIX_TOPIC in filtered:
                print("New GPS fix topic %s replacing old %s" % (GPS_FIX_NEW_TOPIC, GPS_FIX_TOPIC))
                gps_fix_replace = True
            for x in filtered:
                if gps_fix_replace and x == GPS_FIX_TOPIC:
                    # skip old gps topic
                    continue
                self.topic_map[x].append((info['start'], info['path']))
                self.topic_map[x] = sorted(self.topic_map[x])

    def _extend_range(self, start_time, end_time):
        if not self.start_time or start_time < self.start_time:
            self.start_time = start_time
        if not self.end_time or end_time > self.end_time:
            self.end_time = end_time

    def _filter_topics_remap(self, filters):
        return [self._remap_camera[x] if x in self._remap_camera else x for x in filters]

    def write_infos(self, dest):
        for info in self.infos:
            info_path = os.path.splitext(os.path.basename(info['path']))[0]
            write_file = os.path.join(dest, info_path + '.yaml')
            with open(write_file, 'w') as f:
                yaml.dump(info, f)

    def get_message_count(self, topic_filter=[]):
        count = 0
        for info in self.infos:
            if self._remap_camera:
                topic_filter = self._filter_topics_remap(topic_filter)
            filtered = [x['topic'] for x in info['topics'] if not topic_filter or x['topic'] in topic_filter]
            gps_fix_replace = False
            if GPS_FIX_NEW_TOPIC in filtered and GPS_FIX_TOPIC in filtered:
                gps_fix_replace = True
            for topic in info['topics']:
                if ((not topic_filter or topic['topic'] in topic_filter) and
                        (not gps_fix_replace or topic['topic'] != GPS_FIX_TOPIC)):
                    count += topic['messages']
        return count

    def get_readers(self):
        readers = []
        for topic, timestamp_files in iteritems(self.topic_map):
            starts, files = zip(*timestamp_files)
            merged = False
            for r in readers:
                if r.bagfiles == files:
                    r.topics.append(topic)
                    merged = True
            if not merged:
                readers.append(BagReader(bagfiles=files, topics=[topic], remap_camera=self._remap_camera))
        return readers

    def __repr__(self):
        return "start: %s, end: %s, topic_map: %s" % (self.start_time, self.end_time, str(self.topic_map))


def find_bagsets(directory, filter_topics=[], pattern="*.bag"):
    sets = []
    for root, dirs, files in os.walk(directory):
        matched_files = []
        remap_camera = {}
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                if not remap_camera:
                    remap_camera = check_remap_hack(basename)
                filename = os.path.join(root, basename)
                matched_files.append(filename)
        if matched_files:
            set_name = os.path.relpath(root, directory)
            bag_set = BagSet(set_name, matched_files, filter_topics, remap_camera)
            sets.append(bag_set)
    return sets


class BagCursor(object):
    def __init__(self, reader):
        self.latest_timestamp = None
        self.read_count = 0
        self.done = False
        self.vals = []
        self.reader = reader
        self._iter = reader.read_messages()

    def __bool__(self):
        return not self.done

    __nonzero__ = __bool__

    # Advance cursor by one element, store element vals list
    def advance(self, n=1):
        if self.done:
            return False
        try:
            while n > 0:
                topic, msg = next(self._iter)
                self.read_count += 1
                timestamp = msg.header.stamp.to_nsec()
                if not self.latest_timestamp or timestamp > self.latest_timestamp:
                    self.latest_timestamp = timestamp
                self.vals.append((timestamp, topic, msg))
                n -= 1
        except StopIteration:
            self.done = True
        return not self.done

    # Advance cursor by relative time duration in nanoseconds
    def advance_by(self, duration_ns):
        if not self.latest_timestamp and not self.advance():
            return False
        start_time_ns = self.latest_timestamp
        while self.advance():
            elapsed = self.latest_timestamp - start_time_ns
            if elapsed >= duration_ns:
                break
        return not self.done

    # Advance cursor until specified absolute time in nanoseconds
    def advance_until(self, end_time_ns):
        while self.advance():
            if self.latest_timestamp >= end_time_ns:
                break
        return not self.done

    def collect_vals(self, dest):
        dest.extend(self.vals)
        self.vals = []

    def clear_vals(self):
        self.vals = []

    def __repr__(self):
        return "Cursor for bags: %s, topics: %s" % (str(self.reader.bagfiles), str(self.reader.topics))


class CursorGroup(object):
    def __init__(self, readers=[], cursors=[]):
        # a group can be created from readers or existing cursors,
        if readers:
            assert not cursors
            self.cursors = [BagCursor(r) for r in readers]
        elif cursors:
            self.cursors = cursors

    def __bool__(self):
        for c in self.cursors:
            if c:
                return True
        return False

    __nonzero__ = __bool__
    
    def advance(self, n=1):
        all_done = True
        for c in self.cursors:
            if c and c.advance(n):
                all_done = False
        return not all_done

    # Advance all cursors by specified duration
    # Risk of cursors drifting over time from each other
    def advance_by(self, duration_ns=1*SEC_PER_NANOSEC):
        all_done = True
        for c in self.cursors:
            if c and c.advance_by(duration_ns):
                all_done = False
        return not all_done

    # Advance all cursors up to same end time
    def advance_until(self, end_time_ns):
        all_done = True
        for c in self.cursors:
            if c and c.advance_until(end_time_ns):
                all_done = False
        return not all_done

    # Advance the first ready cursor in group by specified amount and bring the reset
    # up to same resulting end time.
    # Risk of pulling in large amounts of data if leading stream has a large gap.
    def advance_by_until(self, duration_ns=1*SEC_PER_NANOSEC):
        all_done = True
        end_time_ns = None
        for c in self.cursors:
            ready = False
            if c:
                if not end_time_ns:
                    ready = c.advance_by(duration_ns)
                    end_time_ns = c.latest_timestamp
                else:
                    ready = c.advance_until(end_time_ns)
            if ready:
                all_done = False
        return not all_done

    def collect_vals(self, dest):
        for c in self.cursors:
            c.collect_vals(dest)
