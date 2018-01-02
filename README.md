
# Caser

A Matlab implementation of Convolutional Sequence Embedding Recommendation Model (Caser) : 

Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding (WSDM '18) 

*Note: I strongly suggest to use the PyTorch version here, as it has better readability and reproducibility.*

# Requirements
* Matlab R2015 + 
* [MatConvNet v1.0](https://github.com/vlfeat/matconvnet)

# Usage
1. Installing MatConvNet ([guide](http://www.vlfeat.org/matconvnet/install/)).
2. Change the [code](https://github.com/graytowne/caser/blob/master/caser_train.m#L2) to make the path point to your MatConvNet path. 
3. Open Matlab and run main_caser.m

# Configurations

#### Data

- Datasets are organized in 2 seperate files: **_train.txt_** and **_test.txt_**

- Same to other data format for recommendation, each file contains a collection of triplets:

  > user, item, rating

  The only difference is the triplets are organized in *time order*.

- As the problem is Sequential Reommendation, the rating doesn't matter, so I convert them to all 1.

#### Model

- <code>L</code>: length of sequence  
- <code>T</code>: number of targets   
- <code>rate_once</code>: whether each item will only be rated once by each user
- <code>early_stop</code>: whether to perform early stop during training
- <code>d</code>: number of latent dimensions   
- <code>nv</code>: number of vertical filters
- <code>nh</code>: number of horizontal filters
- <code>ac_conv</code>: activation function for convolution layer ($\phi_c$)
- <code>ac_fc</code>: activation function for fully-connected layer ($\phi_a$)
- <code>drop_rate</code>: drop ratio when performing dropout

# Comments

For easy implementation and flexibility, I didn't implement below things:

* Didn't make mini-batch in parallel.
* Didn't make the model in MatConvNet [wrapper](http://www.vlfeat.org/matconvnet/wrappers/).
