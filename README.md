# UnetPLUS-OralCancer-image segmentation

Read the ReadMe_OralCancer_imageSegmentation_Documentation.pdf provided in the repo for extensive documentation

Overview
The central thrust of this project is the application of image segmentation techniques to a dataset of oral cancer images sourced from Kaggle. The project employs Convolutional Neural Networks (CNNs), specifically utilizing the U-Net architecture. Unlike typical CNNs, which are employed for image classification tasks, U-Net architectures are particularly potent for image segmentation problems. 

To elevate the performance and efficiency of the U-Net models, various pretrained backbones, namely RESNET34, RESNET101, InceptionNETV3, and EfficientNet, were integrated into the architecture. These pretrained models serve as the encoder in the U-Net architecture, aiding in the extraction of essential features from the images. The decoder then performs the actual segmentation based on these features. 

The Intersection over Union (IOU) score is used as a key performance metric for evaluating the effectiveness of each segmentation model. Given its robustness in evaluating how well the predicted and actual segments overlap, the IOU score is highly valuable in this context. To further optimize the models, a weighted average approach is used to create hybrid models that combine the strengths of individual architectures, thereby aiming to achieve superior IOU scores.

 
