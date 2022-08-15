# Image-based autonomous driving system for an e-fsae

Image-based self-driving algorithm, fully trained in simulation and capable of commanding a vehicle around an unknown track delimited by cones. 

[Video Prototipo andando (Exterior + GAN)]

The following work implements two Neural Networks in tandem, a first one which creates a semantic segmentation and pixel-wise depth estimation of the input image; and a second one, which based on the semantic segmentation outputs the respective vehicle command.

![](resources/architecture.png "architecture")

## Semantic Segmentation and Depth Estimation
To mitigate the existing sim-to-real visual gap, a U-Net is trained to create a segmentation of the cones present in the scene. This converts the current image (belonging to either the real or the simulation domain), into a common simplified domain. To accomplish this, a technique known as Domain Randomization was used.

### Results
The performance of the U-Net over an image belonging to the real domain is shown below.

- [ Input ](https://www.youtube.com/watch?v=FbKLE7uar9Y) (real domain) and segmentation (simplified domain): 

![](resources/gan_segmentation.png "cGAN Segmentation")

- Semantic segmentation of a virtual top view of input image (point-cloud), created with the pixel-wise depth estimation.

![](resources/gan_depth.png "cGAN Segmentation and Depth estimation")

- Semantic segmentation and depth estimation of different scenes.

![](resources/formula_gan.gif "cGAN Segmentation and Depth estimation")

-------------------------

## **Imitation Learning**

![](resources/imitation_3.gif "Racing with a wallfollowing algorithm")

![](resources/imitation_4.gif "Racing with a wallfollowing algorithm")

![](resources/imitation_5.gif "Racing with a wallfollowing algorithm")

