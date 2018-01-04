
# Caser

A Matlab implementation of Convolutional Sequence Embedding Recommendation Model (Caser) from paper: 

*Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18*

*Note: I strongly suggest to use the PyTorch version [here](https://github.com/graytowne/caser_pytorch), as it has better readability and reproducibility.*

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

#### Model Args (in main_caser.m)

- <code>L</code>: length of sequence  
- <code>T</code>: number of targets   
- <code>rate_once</code>: whether each item will only be rated once by each user
- <code>early_stop</code>: whether to perform early stop during training
- <code>d</code>: number of latent dimensions   
- <code>nv</code>: number of vertical filters
- <code>nh</code>: number of horizontal filters
- <code>ac_conv</code>: activation function for convolution layer (i.e., phi_c in paper)
- <code>ac_fc</code>: activation function for fully-connected layer (i.e., phi_a in paper)
- <code>drop_rate</code>: drop ratio when performing dropout

# Citation

If you use this Caser in your paper, please cite the paper:

```
@inproceedings{tang2018caser,
  title={Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding},
  author={Tang, Jiaxi and Wang, Ke},
  booktitle={ACM International Conference on Web Search and Data Mining},
  year={2018}
}
```

# Comments

For easy implementation and flexibility, I didn't implement below things:

* Didn't make mini-batch in parallel.
* Didn't make the model in MatConvNet [wrapper](http://www.vlfeat.org/matconvnet/wrappers/).
