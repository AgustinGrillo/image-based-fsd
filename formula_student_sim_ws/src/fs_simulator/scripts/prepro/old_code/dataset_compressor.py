# load, split and scale the track dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import cv2


# NOTA: Ojo memoria  -->  Habria que usar un generator.
noise_path = '/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/data_set_2/training_set/Noise/'
target_path = '/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/dataset_images/data_set_2/training_set/Target_RGB/'

src_list, tar_list = list(), list()
# enumerate filenames in directory, assume all are images
for dir in listdir(noise_path):
    for filename in listdir(noise_path + dir):
        # load and resize the image
        src_pixels = load_img(noise_path + dir + '/' + filename)
        tar_pixels = load_img(target_path + dir + '/' + listdir(target_path + dir)[0])
        # convert to numpy array
        src_pixels = img_to_array(src_pixels)
        tar_pixels = img_to_array(tar_pixels)
        # crop sky and car
        src_img = src_pixels[184:376, :]
        tar_img = tar_pixels[184:376, :]
        # src_img = src_pixels[184:352, :]
        # tar_img = tar_pixels[184:352, :]
        # # Rescale for pix2pix U-Net compatibility
        # src_img = cv2.resize(src_img, dsize=(512, 256))
        # tar_img = cv2.resize(tar_img, dsize=(512, 256))
        src_list.append(src_img)
        tar_list.append(tar_img)
[src_images, tar_images] = [asarray(src_list), asarray(tar_list)]

print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'dataset_images/dataset_compressed.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)
