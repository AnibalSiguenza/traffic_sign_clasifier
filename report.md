# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First there are a few example of the images with its corresponding label and then an histogram with the frequency of the label appearing in the data set.

![alt text][./examples/dataset_samples.jpg]

![alt text][./examples/data_visualization.jpg]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to add a grayscale layer to the image. Since the signs have also distinctive colors I was not sure if converting the full image to grayscale would lose some valuable information which can be extracted from the colors. Never the less I consider that the easiest way to analyse the shapes of the images is by a gray scale so I added to the images the extra layer giving the input image a shape of (?,36,36,4)

As a last step, I normalized to have a max value of one and centralize the data to have mean equal to cero, since we learn that by doing this the optimization is more stable.

I decided to generate additional data because the data was pretty asymmetric, there were traffic sing which where undersampled. For example, the minimum amount of images is 180, and the maximum is 2010. Consequently while training the undersampled images were underrepresented and this made that the mode did not reach the expected performance. This is why I decided to set at least 500 elements to each label by creating new images based on the already existing ones.

To add more data to the the data set, I used the random zoom method techniques because I read it was a good first try to attempt for a simple dataset. This was successful so I kept it.

Here is an example of an original image and an augmented image:

![alt text][./examples/zoom.jpg]

The difference between the original data set and the augmented data set is a random zoom reason why the sign looks closer in the generated image 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x4 RGB + Grayscale image     			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| Softmax				|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64   				|
| Softmax				|            									|
| Flatten				|												|
| Fully conected		| input 1600 output 400							|
| Softmax				|            									|
| Fully conected		| input 400 output 200							|
| Softmax				|            									|
| Fully conected		| input 200 output 43							|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer for minimizing the softmax cross entropy. The batch size is 128 and the initial number of Epochs of 120. After this number my data set had an oscillating accuracy between 90 and 95, so for having a desired configuration I had an extra retrained epoch of 1 which I ran until the accuracy of both the valid and the test set were at least .93. The learning rate was 0.003

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.941 
* test set accuracy of 0.930

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The original architecture that I used was basically the lenet worked in the class quiz adapted for a color image. It was chosen because the good performance it had with the digits data set.
* What were some problems with the initial architecture?
Basically the accuracy stagnated around .6 accuracy, and I tried changing the hyperparameters, but I concluded that the network was not deep enough to capture all the complexity of the problem. In order words it was underfitting the data. 
* How was the architecture adjusted and why was it adjusted? 
The most important improvement that I observed was the introduction of a grayscale layer. In the preprocessing I added an extra layer of grayscale and adapted the architecture to handle this layer as well. I immediately observed good improvement the accuracy of the validation set went up to .85. So I kept playing around with different number of filters, and Also by replacing the relu by a Softmax I achieved values closed to .9 accuracy.
* Which parameters were tuned? How were they adjusted and why?
Basically I experimented changing all the 3 hyperparameters the epoch number, batch number and learning rate. By changing the batch number the accuracy was always reduced so I keep the "original" 128. The learning rate did make a significant difference and by experimenting I arrived to the value of 0.003 which performs quite well. The number of epochs was increased by a lot compared to the lenet example of the class, and something interesting is that the model oscillates between .90 and .95 accuracy from a single epoch at the final stages this is why I had to create a retrain block in which I could rerun until both the valid and the test set had a accuracy of at least .93 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
It is well known that CNN perform quite good with images since it can recognize patterns of shape throwout the filters. So this was the principal reason to make it the base of this problem. Also as I mentioned before the preprocessing of adding the additional grayscale layer was key in this process, and this was chosen based on my intuition that the grayscale would provide an easier way to find shape patterns, while the rest of the layers could provide important information such as the color, which is important since some traffic signal are read and some are blue for example. The dropout layer can help to prevent the overfitting. I did not use it but it would be interesting if it could help to decreased the overfitting in the training data which in my case has a 1 accuracy, and increase the valid and test data set. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, I resized them to have a proper pixel count for the model:

![alt text][./examples/internet_resized.jpg]

The first image might be difficult to classify because it is not perfectly aligned with the sign and there is something painted below the STOP letters. The second, third and fourth images have a good alignment and contrast with the background so I think it could be well classified. The last image has a bit of noise in front of the signal in the bottom of the sign, this might misleading the classifier. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text]['./examples/internet_predictions.jpg']

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares poorly with the obtain accuracy, but the data sample is too low to give strong conclusion. It is interesting that the one which I though was the most difficult was correctly predicted which is the most noisy and not align stop signal. Also I was expecting the last image to also be predicted correctly, but maybe the noise in front in the bottom was enough to mislead the model

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the Predict the Sign Type for Each Image section of the Ipython notebook.

For the image of Stop  the top predictions were:
   Stop  probability = 1.000
   Priority road  probability = 0.000
   Speed limit (80km/h)  probability = 0.000
   No entry  probability = 0.000
   Speed limit (60km/h)  probability = 0.000
For the image of General caution  the top predictions were:
   General caution  probability = 1.000
   Traffic signals  probability = 0.000
   Wild animals crossing  probability = 0.000
   Road work  probability = 0.000
   Double curve  probability = 0.000
For the image of Wild animals crossing  the top predictions were:
   Wild animals crossing  probability = 0.999
   Slippery road  probability = 0.000
   Double curve  probability = 0.000
   Dangerous curve to the left  probability = 0.000
   Dangerous curve to the right  probability = 0.000
For the image of Turn right ahead  the top predictions were:
   Turn right ahead  probability = 1.000
   Go straight or left  probability = 0.000
   Keep left  probability = 0.000
   No passing  probability = 0.000
   Ahead only  probability = 0.000
For the image of Speed limit (60km/h)  the top predictions were:
   Traffic signals  probability = 0.586
   Speed limit (60km/h)  probability = 0.200
   Dangerous curve to the left  probability = 0.098
   General caution  probability = 0.081
   Keep left  probability = 0.015
   
As we can see in the first 4 cases the prediction was strongly leaning into the correct prediction with almost 100% of probably. The only case in which the probably is more distributed was in the incorrect classified image of Speed limit (60km/h) which ended up being the second place with .2 after the incorrect .586 given to Traffic signals

