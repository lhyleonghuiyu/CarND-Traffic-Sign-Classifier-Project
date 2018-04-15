# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/3_60mphh.jpg "Traffic Sign 1"
[image5]: ./examples/4_70mph.png "Traffic Sign 2"
[image6]: ./examples/21_doublecurve.png "Traffic Sign 3"
[image7]: ./examples/22_bumpy.png "Traffic Sign 4"
[image8]: ./examples/39_keepleft.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1.Summary of data set In the code

In[68] and In[53] contain the code used to generate the summary statistics of the traffic signs data set. The following is a brief summary of the data set: 

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3, the 3 channels storing RGB information
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

As an exploration of the data set,  histograms of the training, validation and test set were plot to observe the distribution of the data. Out[53] contains a histogram of the training set, validation set and test set in the aforementioned order. An interesting observation is that the frequency distribution of the output classes seem to be rather similar in all three sets of data. 

### Design and Test a Model Architecture

#### 1. Image Pre-processing

As a first step, I decided to convert the images to grayscale to reduce the amount of information being fed to the neural network. The traffic signs may be identified primarily based on their shapes and figures without color by the human eye, hence in a similar sense, feeding this information to the neural network would reduce the amount of information being processed, thereby saving time and also improving the accuracy of the network. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data values to between -0.5 and 0.5 as this would be the optimal values in order for the LeNet architecture to work effectively. 


#### 2.Final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        	| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image  		|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6  	| 
| RELU 			| RELU	Activation function			|
| Max pooling 2x2      	| 2x2 stride,  outputs 14x14x6			| 
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16   |
| RELU    		| RELU Activation Function			|
| Max pooling 2x2	| 2x2 stride, outputs 5x5x16			|
| Fully Connected Layer | 5x5x16, outputs 256				|
| Fully Connected Layer | outputs 120 with dropout of 0.5		|
| Fully Connected Layer | outputs n_classes				|

The code for the neural net architecture may be found in code cell In[96].


#### 3. Training parameters

To train the model, the following parameters were used: 
* Adam Optimizer: The gradient descent used to optimize the weights trained rather well, hence no changes were made to this from the base code. 
* Batch size of 128: The data set was too large and therefore had to be broken down into smaller sets. At the same time, we would not want the batch size to be too small as this would result in multiple iterations 
* 1400 epochs: With the dropout layer, the training took a larger number of epochs to converge.
* Learning rate of 0.001: Based on tuning after several attempts between 0.001, 0.002, 0.005, and 0.01, it was found that the learning rate of 0.001 resulted in the best accuracy out of the three values. 


#### 4. Chosen Architecture and Approach 

I used the LeNet architecture which was proposed by Yann Lecun, as it has been proven to be a highly accurate image classifier when trained over a large number of epochs with tuned hyperparameters. 

However, the parameters in the fully connected layers were modified slightly from the neural net architecture provided in the deep learning lectures: 

1. A dropout layer was implemented to retain the weights that were perceived to be relevant. While this network architecture took a larger number of epochs to converge, the addition of a single droupt layer at the second last layer improved the accuracy by approximately 3%. 
2. The number of neurons in each fully connected hidden layer was increased to allow the neural net to detect a greater number of features from the images, before deciding whether these features were relevant in the final classification or not through the use of the dropout layer. Detecting more featres would allow a better regression fit in the final layer. 

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 94.5%
* test set accuracy of 92.6%

It was observed that possibly due to the dropout layer, the convergence proved more to be a range of accuracies, rather than a single accuracy value, which was observed when the dropout layer was not implemented originally. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			|     Prediction		       		| 
|:---------------------:|:---------------------------------------------:| 
| 60kmh		      	| 60kmh						| 
| 70kmh     		| U-turn					|
| Doublecurve		| General Caution				|
| Bumpy Road	      	| Bumpy Road 					|
| Keep Left		| Keep Left 					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92.6%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in code cell In[76] of the notebook. 

The table below shows the top 5 softmax predictions based on probabilities output from the model for each image. 

|	Image      	|     Prediction (Probability)		| 
|:---------------------:|:-------------------------------------:| 
| 	60kmh  		|  60kmh(1)				| 
| 	70kmh		| 30kmh (0.645), Traffic Signals (0.218), Pedestrians (0.136), 20kmh (1.93e-3), General Caution (2.94e-5)		|
| Double curve 		| General Caution (0.998), Right-of-way at next intersection (1.83e-4), Slippery road (1.61e-7), Beware of ice and snow (1.4e-22), Children Crossing (7.11e-26) | 	
|  Bumpy Road		| Bumpy Road (1)			|	
| Keep Left 		| Keep Left (1), Turn Right ahead (6.14e-15)	|



