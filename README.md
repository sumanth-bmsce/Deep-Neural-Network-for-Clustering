# Deep-Neural-Network-for-Clustering
Autoencoders -  a deep neural network was used for feature extraction followed by clustering of the "Cancer" dataset using k-means technique

<h3> Objective </h3>
 This project is an attempt to use “Autoencoders” which is a non-linear dimensionality reduction technique for feature extraction and then  use the hidden layer activations which is given as input to the k-means algorithm for clustering. 
 
 ![ds](https://github.com/sumanth-bmsce/Deep-Neural-Network-for-Clustering/blob/master/r_arch.png)
 
 <h3> Modules </h3>
 This project has two main components:

1.	**Autoencoders** : In this module, the objective is to give the .csv file as input to the input layer, get the hidden layer activations from the hidden layer. This is done using the gradient descent algorithm. The loss function used is the cross entropy loss function. The hidden layer activations are given as input to kmeans algorithm for clustering.

2.	**K-means** : Linearly clustering the input where the input comes from the autoencoders and displaying the confusion matrix and clustering accuracy.

<h3> Algorithm </h3> 

**Autoencoders**

**Input** : Input data matrix, No of hidden neurons, Weight matrix(W), No of clusters for k-means.

Let : 

•	X is the input data</br>
•	Y is the hidden layer activations</br>
•	Z is the predicted output or the reconstruction of the input X.</br>
•	W denote the weights from input to hidden layer</br>
•	b is the input and hidden layer bias</br>
•	s(.) denote the sigmoidal function</br>

1.	Take the input X ε [0,1] and map it ( with an encoder )  to a hidden representation y ε [0,1] through a deterministic mapping.

2.	The latent representation , or code is then mapped back (with a decoder) into a reconstruction  of the same shape as . The mapping happens through a similar transformation.
	
3.	The reconstruction error is calculated using the cross- entropy loss function.

4.	The weights are updated using the gradient descent equation.

**K-means Clustering : **

5.	Initialize the centroids randomly.
6.	Update the centroids based on the Eucledian distance.
7.	Group the datapoints based on minimum distance.
8.	Perform steps 5,6,7 for a certain number of iterations.

**Output** : Confusion Matrix and Clustering Accuracy

<h3> Results Screenshots </h3>





