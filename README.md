# Facial Keypoints Detection
### Don Moon, Lester Yang, Shwetha Chitta Nagaraj
### W207 – Spring 2020

### Goal: 

The challenge here is to detect facial keypoint locations on images from the dataset provided by [Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/overview).
The team explored the training dataset to understand the features(images - 96X96) and labels(30 keypoints x, y) to predict. Exploration of basic neural network and popular Convolutional Neural Networks(CNN) are done to apply these concepts in designing and training a model to make predictions on the given test data. The predictions are submitted to Kaggle which then uses Root Mean Squared Error(RMSE) to evaluate the predictions. 

### Navigating the repository

- **FKP_Team_Baseline-Colab.ipynb** - This notebook has the data exploration of the Kaggle dataset and estimating a baseline for orientation. This notebook was used for the baseline presentation. 
- **FKP_Initial_Models_Exploration.ipynb** - This notebook explores the world of neural networks starting with a 1-layer neural network and goes on to explore and model over some classical and popular architectures like LeNet-5, AlexNet, VGG-16. The idea is get an idea of the different characteristics, structure, parameters comprising such architectures and their performance on the kaggle training data. 
- **FKP_Initial_Models_with_Split_Data.ipynb** - This notebook again explores Convolutional Architectures used in the 2nd notebook by training on them on split training datasets. The team will use dataset 1 to train 4 keypoints and 2nd dataset will have 11 other keypoints. This split is also explained in the notebook. The performance of the models on the split data are evaluated against Root Mean Squared Error(RMSE) which is the std. set by Kaggle for this challenge. The best model from here will be used in next steps.
- **FKP_DataAug_SimpleCNN_VGG16_Evaluation.ipynb** - This explores data transformation techniques like mirroring, rotation, adjusting contrast in images and blurring(random noise) to the images. Each one of the techniques are applied to the split datasets and a Simple CNN will be trained to compare performances. The best transformation technique or combination will then be applied on the VGG-16 condensed model explored in the initial models exploration phase. It was found that the VGG model underperformed using by large GPU resources and long training times due to the millions of parameters it has to train. The team though scored a high kaggle score without the augmented data, could not keep up with the expanded datasets. 
- **FKP_DataAug_Final_Models.ipynb** - In this notebook, we take the heavy weight VGG-16 model and simplify it with smaller filter depths while maintaining the the same number of layers. This is trained against the augmented datasets and evaluated based on RMSE, Accuracy of predictions and Kaggle submission score. Further improvements were made to increase the performance of the model and with this several variations of the model were explored and performance evaluated. The Kaggle submission scores gives details as which one performed the best and scored well on the Kaggle Leaderboard.
- **FKP_leveraing_pretrained_models.ipynb** - In this notebook, three pre-trained models, MobileNet, ResNet, and InceptionNet are leveraged and augmented for the Facial Keypoint Detection Kaggle competition.
- **pre-trained_models/** - This folder contains the MobileNets and InceptionNets trained for Facial Keypoint Detection and their Kaggle scores
- **FKP_Facebox_detect.ipynb** - Implementation of a model pipeline, where a pre-trained face detector [MTCNN](https://github.com/ipazc/mtcnn) based on cascaded convolutional networks is used for detecting faces, cropping, and resizing. Missing labels and wrong labels can also be imputed at this stage. The new images are then fed into a second CNN with the VGG-16 architechture for detecting facial keypoints. Unfortunately, this implementation was not trained due to time limit.

- **W207_Final_Project_Presentation.pdf** - Final presentation slides
