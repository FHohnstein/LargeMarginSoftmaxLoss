# Large Margin Softmax Loss for Convolutional Neural Networks <br/>  <font size="6"> An Implementation in Pytorch </font> 

<br/>

This project attempts to implement the [Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1612.02295.pdf)
proposed by Weiyang Liu, Yandong Wen, Zhiding Yu and Meng Yang.

## Setup

In order to use this code base, install all requirements using pip and 
python **3.8** or newer versions.


```console
pip3 -r install requiremetns.txt
```

### Using [virutalenv](https://pypi.org/project/virtualenv/)
This project was developed using virtualenv. Install virtualenv using pip3.
```console
pip3 install virtualenv 
```
Next, create a virtual-environment named *interpreter*.
```console
python -m virtualenv --python=/usr/bin/python3.8 interpreter
```
Activate the environment and upgrade pip.
```console
source interpreter/bin/activate
pip3 install --upgrade pip
```
Finally, install all dependencies listed inside the requirements.txt.
```console
pip3 -r install requirements.txt
```

## Usage
After cloning the project and installing all requirements:

* to see an example of using the large-margin softmax loss, run ..
	```console
	cd imbalanced_cifar10/new_loss/
	python large_margin_softmax.py
	```

* to train with
  * balanced CIAFAR10 and CrossEntropyLoss and
    * ResNet18
       ```console 
        cd balanced_cifar10/with_resnet18
        python train_resnet18.py
       ```
    * VGG16
       ```console 
        cd balanced_cifar10/with_vgg16
        python train_vgg16.py 
       ```
  * imbalanced CIAFAR10
    * CrossEntropyLoss
      * ResNet18
        ```console 
        cd imbalanced_cifar10/baseline/with_resnet18
        python train_resnet18.py
        ```
	  * VGG16 
        ```console 
        cd imbalanced_cifar10/baseline/with_vgg16
        python train_vgg16.py
        ```
    * <span style="color:green">LargeMarginSoftmaxLoss</span>
      * ResNet18
        ```console 
        cd imbalanced_cifar10/new_loss/with_resnet18
        python train_resnet18.py
        ```
	  * VGG16
        ```console 
        cd imbalanced_cifar10/new_loss/with_vgg16
        python train_vgg16.py
        ```
<br/>

## Hyperparameter

To adjust the hyperparameter to your liking, change the values 
within the header of the file. 

<br/>

```python
# -------------------- Parameter -------------------- #

SEED = 42
torch.manual_seed(SEED)

IMBALANCE_TYPE = "exp"
IMBALANCE_FACTOR = 0.1

USE_CUDA = True

N_EPOCHS = 10
BATCH_SIZE = 32
TRAIN_VALIDATION_SPLIT = 0.9
LOG_ON_DIFF_TOL = 0.01

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

LEARN_RATE = 0.001
LEARN_RATE_STEP_SIZE = 2
LEARN_RATE_GAMMA = 0.9

# ----------------------- Utils ---------------------- #
```
      

<br/>




## Project Structure

* imbalance_cifar.py
  * Implementation of the [imbalanced CIFAR10](
  https://github.com/kaidic/LDAM-DRW) dataset by Kaidi Cao  and Bhavin Jawade
* large_margin_softmax.py
  * Implementation of the Large-Margin Softmax Loss
  * Contains an example Use-Case
* train_resnet18.py
  * Train ResNet18 using the dataset and loss, according to the project structure
* train_vgg16.py
  * Train Vgg16 using the dataset and loss, according to the project structure

<br/>

```tree  * Example Use-Case
├── balanced_cifar10
	    ├── with_resnet18
	    │   └── train_resnet18.py
	    │
	    └── with_vgg16
	        └── train_vgg16.py

├── imbalanced_cifar10
	├── imbalance_cifar.py
    │
    ├── baseline
    │   ├── with_resnet18
    │   │   └── train_resnet18.py
    │   │ 
    │   └── with_vgg16
    │       └── train_vgg16.py
    │   
    └── new_loss
        ├── large_margin_softmax.py
        │
        ├── with_resnet18
        │   └── train_resnet18.py
        │
        └── with_vgg16
            └── train_vgg16.py
```
  
