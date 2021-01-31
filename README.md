 # Study of different CNN Architectures
 
 I have implemented various CNN Architecture on Image Classification datasets and also implemented YOLO Algorithm and Faster RCNN  for object detetion problem.
There are various architectures of CNNs available which have been key in building algorithms which power and shall power AI as a whole in the foreseeable future.
Some of the CNN Architecture which i have implemented in this Project are:

    1. AlexNet
    2. VGG16/VGG19
    3. ResNet
    4. Inception Network
    5. EfficientNet
    
For Object Detection ,I have implemented 

     1.> Faster RCNN Algo
     2.> Yolo Algorithm

# AlexNet :  
   AlexNet was primarily designed by Alex Krizhevsky. It was published with Ilya Sutskever and Krizhevsky’s doctoral advisor Geoffrey Hinton, and is a                Convolutional Neural Network or CNN.After competing in ImageNet Large Scale Visual Recognition Challenge, AlexNet shot to fame. It achieved a top-5                error of 15.3%. This was 10.8% lower than that of runner up. 

The primary result of the original paper was that the depth of the model was absolutely required for its high performance. This was quite expensive             computationally but was made feasible due to GPUs or Graphical Processing Units, during training.

I have implemented this CNN architecture on PlantVillage Dataset .It contains dataset of diseased plant leaf images and corresponding labels.

# VGG :
   VGGNet is invented by Visual Geometry Group (by Oxford University). This architecture is the 1st runner up of ILSVR2014 in the classification task while           the winner is GoogLeNet. The reason to understand VGGNet is that many modern image classification models are built on top of this architecture. Just like         word2vec in NLP field.This story will discuss Very Deep Convolutional Networks for Large-scale Image Recognition (Simonyan et al., 2014).

It is currently the most preferred choice in the community for extracting features from images. The weight configuration of the VGGNet is publicly available and has been used in many other applications and challenges as a baseline feature extractor.

I have implemented this CNN architecture on Flower Recognition Dataset . This dataset contains 4242 images of flowers.The pictures are divided into five classes: chamomile, tulip, rose, sunflower, dandelion.

# Inception :
   Going deeper with convolutions, Szegedy et al. (2014), this paper introduces the Inception v1 architecture, implemented in the winning ILSVRC 2014                submission GoogLeNet. The main contribution with respect to Network in Network is the application to the deeper nets needed for image classification.              From a theoretical point of view, Google's researchers observed that some sparsity would be beneficial to the network's performance, and implemented              it using today's computing techniques.
   
After the initial breakthrough of the Inception architecture, most changes have been incremental. This means that any of these models would be good enough for initial work in new research areas outside of classification.Inception-v3 pretrained weights are widely available, both in Keras and Tensorflow.

I have implemented this CNN architecture on Chest X-Ray Images (Pneumonia) . here are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).


# ResNet: 
   ResNet, which was proposed in 2015 by researchers at Microsoft Research introduced a new architecture called Residual Network.Since ResNet blew people’s           mind in 2015, many in the research community have dived into the secrets of its success, many refinements have been made in the architecture.  Deep               Residual Network was arguably the most groundbreaking work in the computer vision/deep learning community in the last few years.

In order to solve the problem of the vanishing/exploding gradient, this architecture introduced the concept called Residual Network. In this network we use a technique called skip connections . The skip connection skips training from a few layers and connects directly to the output.

I have implemented this CNN architecture on Chest X-Ray Images (Pneumonia) . here are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

# EfficientNet : 
   In May 2019, two engineers from Google brain team named Mingxing Tan and Quoc V. Le published a paper called “EfficientNet: Rethinking Model                      Scaling for Convolutional Neural Networks”. The core idea of publication was about strategically scaling deep neural networks but it also                          introduced a new family of neural nets, EfficientNets.
             
The scaling method introduced in paper is named compound scaling and suggests that instead of scaling only one model attribute out of depth,                       width, and resolution; strategically scaling all three of them together delivers better results.

The base model of EfficientNet family, EfficientNet-B0.The EfficientNet-B0 architecture wasn’t developed by engineers but by the neural network itself. They developed this model using a multi-objective neural architecture search that optimizes both accuracy and floating-point operations.

I have implemented this CNN architecture on SIIM-ISIC Melanoma Classification.I am predicting a binary target for each image. Your model should predict the probability (floating point) between 0.0 and 1.0 that the lesion in the image is malignant (the target). 


![oie_JiStYAeu4q31](https://user-images.githubusercontent.com/62735655/106376951-8834d800-63bf-11eb-9370-73bcd99af3ea.png)



Object Detection :

# Faster RCNN : 
   The most widely used state of the art version of the R-CNN family — Faster R-CNN was first published in 2015. This article, the third and final one of a          series to understand the fundamentals of current day object detection elaborates the technical details of the Faster R-CNN detection pipeline.
         
In the R-CNN family of papers, the evolution between versions was usually in terms of computational efficiency ,reduction in test time, and improvement in performance (mAP).These networks usually consist of — a) A region proposal algorithm to generate “bounding boxes” or locations of possible objects in the image; b) A feature generation stage to obtain features of these objects, usually using a CNN; c) A classification layer to predict which class this object belongs to; and d) A regression layer to make the coordinates of the object bounding box more precise.

I have implemented Faster RCNN  on Custom Dataset that i have created and submitted the code for it.

# YOLO Algo :
   YOLO (You Only Look Once) real-time object detection algorithm, which is one of the most effective object detection algorithms that also encompasses many          of the most innovative ideas coming out of the computer vision research community.
   YOLO came on the computer vision scene with the seminal 2015 paper by Joseph Redmon et al. “You Only Look Once: Unified, Real-Time Object Detection,” and          immediately got a lot of attention by fellow computer vision researchers.
       
       
       
# Thought behind this Project :
   In order to get started into the field of Image Processing or to improve the prediction accuracy of the custom CNN models, the knowledge of                        some of the famous CNN architectures will keep us reach the pace in this competitive and ravenous world.in this project I will be covering                        a few famous CNN architectures. I mainly stress on how to implement them in Keras in general i.e., making custom models with the idealogy                          and structure of those architectures.














