# ThyNet
This repository contains source code for the paper "Deep learning-based artificial intelligence assists in thyroid nodule management: a multi-center, diagnostic study"

# Citation
If you find DenseNet useful in your research, please consider citing:
"Deep learning-based artificial intelligence assists in thyroid nodule management: a multi-center, diagnostic study"

# Introduction
ThyNet is designed for malignant thyroid classification. It is essentially a ensemble of three sub-networks which use ResNet, DenseNet and ResNext as backbones,respectively.

# Usage
## Depend on the environment
pytorch>=1.5.0
torchvision>=0.6.0
numpy>=1.17.0
pillow>=7.1.2
opencv>=4.2.0
scikit-learn>=0.23.1
matplotlib>=3.2.1

## Train processing
### 1. Prepare the data
   
You can use code to automatically divide the training set from the test set or by yourself.

if you using code, you need to put the data in the "data" directory in the following structure

--data

    --your dataset

        --B
            img1.jpg
            img2.jpg
            img3.jpg
             ......
        --M 
            img4.jpg
            img5.jpg
            img6.jpg
             ......
And, If you divide up your training set and your test set beforehand, You need to put the data in the "data" directory in the following structure
  

--data

    --train dataset

        --B
            img1.jpg
            img2.jpg
            img3.jpg
             ......
        --M 
            img4.jpg
            img5.jpg
            img6.jpg
             ......
    --test dataset

        --B
            img1.jpg
            img2.jpg
            img3.jpg
             ......
        --M 
            img4.jpg
            img5.jpg
            img6.jpg
             ......

### 2. Training step
**python train.py --batch-size 32 --model resnet101 --gpu 0 --num_class 2 --split_train_ratio 0.8 --task_name thynet --path data --auto_split 1**

--batch_size  the training batch size

--model 
     
        Choose "resnet101" to train resnet

        Choose "densenet201" to train densenet
  
        Choose "resnext101" to train resnext

--gpu Using gpu id, mutil gpu can use "0,1,2,3"

--num_class The number of categories of tasks. ThyNet is 2.

--split_train_ratio Automatically partition the proportion of the training set. It only works if the partition is automatic. The default is 0.8.

--path Data storage path

--auto_split Whether to automatically divide training sets and test sets. "1" is spilt and "0" is not.


### 3. Testing step
**python test.py --resnet_model_path resnet_model_path --densenet_model_path densenet_model_path --resnext_model_path resnext_model_path --imgpath test.jpg --gpu 0 --num_class 2**

--resnet_model_path Trained ResNet model path

--densenet_model_path Trained DenseNet model path

--resnext_model_path Trained ResNext model path

----imgpath Test the image path

--gpu Using gpu id, mutil gpu can use "0,1,2,3"

--num_class The number of categories of tasks. ThyNet is 2.
