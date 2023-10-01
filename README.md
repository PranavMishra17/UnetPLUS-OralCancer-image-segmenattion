# UnetPLUS-OralCancer-image segmentation

Read the ReadMe_OralCancer_imageSegmentation_Documentation.pdf provided in the repo for extensive documentation

<div style="text-align: center; background-color: white;">
  <figure>
    <img src="https://github.com/PranavMishra17/UnetPLUS-OralCancer-image-segmenattion/blob/main/Model%20predictions%20and%20metrics/_Hybrid%20models_/Prediction/w1.png?raw=true" alt="Final comparision between ground truth and model predicted segmentation of Oral Cancer image" style="background-color: white;">
    <figcaption style="background-color: white;"><i>Final comparision between ground truth and model predicted segmentation of Oral Cancer image</i></figcaption>
  </figure>
</div>


![Final comparision between ground truth and model predicted segmentation of Oral Cancer image](./Model%20predictions%20and%20metrics/_Hybrid%20models_/Prediction/w1.png)
**Caption**: Final comparision between ground truth and model predicted segmentation of Oral Cancer image


**Overview**
The central thrust of this project is the application of image segmentation techniques to a dataset of oral cancer images sourced from Kaggle. The project employs Convolutional Neural Networks (CNNs), specifically utilizing the U-Net architecture. Unlike typical CNNs, which are employed for image classification tasks, U-Net architectures are particularly potent for image segmentation problems.

To elevate the performance and efficiency of the U-Net models, various pretrained backbones, namely RESNET34, RESNET101, InceptionNETV3, and EfficientNet, were integrated into the architecture. These pretrained models serve as the encoder in the U-Net architecture, aiding in the extraction of essential features from the images. The decoder then performs the actual segmentation based on these features.

The Intersection over Union (IOU) score is used as a key performance metric for evaluating the effectiveness of each segmentation model. Given its robustness in evaluating how well the predicted and actual segments overlap, the IOU score is highly valuable in this context. To further optimize the models, a weighted average approach is used to create hybrid models that combine the strengths of individual architectures, thereby aiming to achieve superior IOU scores.


**Additional Notes on Project Environment and Code**

Google Collab as the Development Environment 

For this project, Google Collab was chosen as the development environment due to its ease of use and availability of substantial computational resources, including GPUs. The cloud-based setting made it convenient to access and run the code from different locations and devices, offering flexibility during the development phase. 

Data and Model Availability 

It's important to note that the GitHub repository hosting this code does not contain the dataset used for the project, nor does it host any of the trained models or predictions. While the code serves as a reference, it's not intended for public use of models or predictions. However, some example outputs and graphs are provided both in the repository and in the document to offer a level of understanding of the work conducted. 

Code Documentation 

The code lacks extensive documentation, and some sections are deprecated or not in use as the project evolved over time. While comments have been added to facilitate understanding, the code itself is not as polished or well-documented as one might expect for a public release. This makes the present documentation all the more crucial for anyone looking to understand the procedures, methodologies, and outcomes detailed in this project. 
 
