# Image-based autonomous driving system for an e-fsae

Image-based self-driving algorithm, fully trained in simulation and capable of commanding a vehicle around an unknown track delimited by cones. 

[Video Prototipo andando (Exterior + GAN)]

The following work implements two Neural Networks in tandem, a first one which creates a semantic segmentation and pixel-wise depth estimation of the input image; and a second one, which based on the semantic segmentation outputs the respective vehicle command.

[Esquema filmina 67]

## Semantic Segmentation and Depth Estimation
To mitigate the existing sim-to-real visual gap, a U-Net is trained to create a segmentation of the cones present in the scene. This converts the current image (belonging to either the real or the simulation domain), into a common simplified domain (M). To accomplish this, a technique known as Domain Randomization was used.

-------------------------

## **Imitation Learning**

![](resources/imitation_3.gif "Racing with a wallfollowing algorithm")

![](resources/imitation_4.gif "Racing with a wallfollowing algorithm")

![](resources/imitation_5.gif "Racing with a wallfollowing algorithm")

## **Perception (cGAN)**
![](resources/formula_gan.gif "cGAN Segmentation and Depth estimation")

