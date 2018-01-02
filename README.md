
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

**Data**

- Datasets are organized in 2 seperate files: train.txt and test.txt

- Same to other data format for recommendation, each file contains a collection of triplets:

  user, item, rating

  The only difference is the triplets are organized in time order.

- As the problem is Sequential Reommendation, the rating doesn't matter, so I convert them to all 1.

**Model**

- Datasets are organized in 2 seperate files: train.txt and test.txt
- ​

# Comments

For easy implementation and flexibility, I didn't implement the above things:

* Didn't make mini-batch in parallel.
* Didn't make the model in MatConvNet [wrapper](http://www.vlfeat.org/matconvnet/wrappers/).

