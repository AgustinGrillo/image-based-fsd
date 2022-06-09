import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2

tf.compat.v1.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras
tf.compat.v1.keras.backend.set_session(sess)


# Load model
generator_model = tf.keras.models.load_model('test_train_pix2pix/test9/generator_model_epoch_190')

# Load image
# test_img_path = 'dataset_images/data_set_2/training_set/Noise/mid22/mid22_6.jpg'
test_img_path = 'dataset_images/real_test_images/test_9_1.jpg'
test_video_path = 'dataset_images/real_test_images/AMZ_driverless.mp4'


# VIDEO EVALUATION

cap = cv2.VideoCapture(test_video_path)

while(cap.isOpened()):
    ret, frame = cap.read()
    img = frame[250:442, :640, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_3d = img.astype(np.float32)
    img_3d = img_3d / 255.0
    pts_1 = [[146,155],  [115,192], [390,192], [264, 94]]
    pts_2 = [[0, 0], [0, 40], [320, 0]]
    pts_3 = [[640, 0], [320, 0], [640, 40]]
    img_3d = cv2.fillPoly(img_3d, np.array([pts_1]), (0.5, 0.5, 0.5))
    img_3d = cv2.fillPoly(img_3d, np.array([pts_2]), (0.5, 0.5, 0.5))
    img_3d = cv2.fillPoly(img_3d, np.array([pts_3]), (0.5, 0.5, 0.5))
    img_3d = 2 * img_3d - 1.0
    img = np.expand_dims(img_3d, 0)

    gen_img = generator_model(img, training=False)
    sim = gen_img[0][0, :, :, :]
    depth = gen_img[1][0, :, :, 0]

    sim_np = sim.numpy()
    depth_np = depth.numpy()
    input_img = img_3d


    sim_np = 0.5 * (sim_np + 1.0)
    depth_np = 0.5 * (depth_np + 1.0)
    input_img = 0.5 * (img_3d + 1.0)

    sim_np = cv2.cvtColor(sim_np, cv2.COLOR_RGB2BGR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    cv2.imshow('original', input_img)
    cv2.imshow('sim', sim_np)
    cv2.imshow('depth', depth_np)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# # IMAGE EVALUATION
# img = cv2.imread(test_img_path)
# img = img.astype(np.float32)
# # img = img[184:376, :]
# # img = img[76:268, :]
# img = np.expand_dims(img, 0)
# img = img/255.0
#
# for _ in range(100):
#     start = time.time()
#     gen_img = generator_model(img, training=False)
#     print("[DEBUG] Time taken to generate images: {}".format(time.time()-start))
#     start = time.time()
#     sim = gen_img[0][0, :, :, :]
#     depth = gen_img[1][0, :, :, 0]
#     sim_np = sim.numpy()
#     sim_np = cv2.cvtColor(sim_np, cv2.COLOR_RGB2BGR)
#     depth_np = depth.numpy()
#     print("[DEBUG] Time taken to convert images: {}".format(time.time() - start))
#     start = time.time()
#     # plt.imshow(sim)
#     # plt.pause(0.0001)
#     cv2.imshow('img_sim', sim_np)
#     # cv2.imshow('img_depth', depth_np)
#     cv2.waitKey(1)
#     print("[DEBUG] Time taken to plot images: {}".format(time.time() - start))
