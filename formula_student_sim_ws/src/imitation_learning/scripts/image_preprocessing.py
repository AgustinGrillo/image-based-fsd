# image pre-processing script.
# Image in cv2 format (no ROS)

import numpy as np
import cv2


class process:

    def show_window(self, title, img):
        cv2.imshow(title, img)

    def load_image(self, path):
        img = cv2.imread(path)
        return img

    def cones2bw(self, img):

        # Input: image (img) in BGR format.
        # Output: image (img_mod) with black and white colored cones in BGR format.

        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_mod_HSV = np.copy(img_HSV)

        # blue: HUE = [110, 125]  in open cv hue range [0:180]
        # yellow: HUE = [25, 33]

        log1 = img_mod_HSV[:, :, 0] > 100
        log2 = img_mod_HSV[:, :, 0] < 130
        logs1 = img_mod_HSV[:, :, 1] > 50  # 0.18 * 255
        # logs2 = img_mod_HSV[:, :, 1] < 127
        logv1 = img_mod_HSV[:, :, 2] > 50  # 0.35 * 255
        # logv2 = img_mod_HSV[:, :, 2] < 127
        log12 = log1 * log2 * logs1 * logv1  # blue cones position

        log3 = img_mod_HSV[:, :, 0] > 20
        log4 = img_mod_HSV[:, :, 0] < 38
        logs3 = img_mod_HSV[:, :, 1] > 50  # 0.15 * 255
        # logs4 = img_mod_HSV[:, :, 1] < 127
        logv3 = img_mod_HSV[:, :, 2] > 50  # 0.5 * 255
        # logv4 = img_mod_HSV[:, :, 2] < 127
        log34 = log3 * log4 * logs3 * logv3  # yellow cones position

        # we will paint blue cones of white, and yellow cones of black (modifying the HSV values).
        # blue cones to white
        # img_mod_HSV[:, :, 0] = np.where(log12, 0, img_mod_HSV[:, :, 0])
        img_mod_HSV[:, :, 1] = np.where(log12, 0, img_mod_HSV[:, :, 1])
        img_mod_HSV[:, :, 2] = np.where(log12, 255, img_mod_HSV[:, :, 2])
        # yellow cones to black
        # img_mod_HSV[:, :, 0] = np.where(log34, 0, img_mod_HSV[:, :, 0])
        img_mod_HSV[:, :, 1] = np.where(log34, 0, img_mod_HSV[:, :, 1])
        img_mod_HSV[:, :, 2] = np.where(log34, 0, img_mod_HSV[:, :, 2])

        img_mod = cv2.cvtColor(img_mod_HSV, cv2.COLOR_HSV2BGR)

        return img_mod

    def img2gray(self, img):
        img_mod = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_mod

    def smooth_img(self, img, smooth_type='erosion'):
        # Types: erosion, opening, blur
        kernel = np.ones((5, 5), np.uint8)

        if smooth_type == 'erosion':
            img_mod = cv2.erode(img, kernel, iterations=1)
        elif smooth_type == 'opening':
            img_mod = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  #### puede servir ####
        elif smooth_type == 'blur':
            img_mod = cv2.GaussianBlur(img, (5, 5), 0)
        else:
            print('Smoothing type error.')
            img_mod = None

        return img_mod

    def pixelate_img(self, img, height_pixelation=4, width_pixelation=4):
        height, width = img.shape[:2]

        img_pixelated = cv2.resize(img, (int(width / width_pixelation), int(height / height_pixelation)),
                                   interpolation=cv2.INTER_LINEAR)
        img_pixelated_scaled = cv2.resize(img_pixelated, (width, height), interpolation=cv2.INTER_NEAREST)

        return img_pixelated, img_pixelated_scaled

    def destroy_windows(self):
        cv2.waitKey(0)

    ##    cv2.destoyAllWindows()

    def process_img(self, img, pixelation, screen_size=[170, 640]):

        processed_image = img[:, :, 0]  # blue channel
        processed_image = processed_image[185:185 + screen_size[0],
                          (640 - screen_size[1]) / 2:(640 - screen_size[1]) / 2 +
                                                     screen_size[1]]  # crop: [185:355, :]
        processed_image, processed_image_scaled = self.pixelate_img(processed_image, pixelation, pixelation)  # pixelate
        return processed_image, processed_image_scaled
