# Attention Guided Deep Features for Accurate Body Mass Index Estimation

Body Mass Index (BMI) has been widely used as an indicator to evaluate the health condition of individuals, classifying a person as underweight, normal weight, overweight, or obese. Recently, several methods have been proposed to obtain BMI values based on the visual information, e.g., face images or 3D body images. These methods by extrapolating anthropometric features from face images or 3D body images are advanced in BMI estimation accuracy, however, they suffer from the difficulties of obtaining the required data due to the privacy issue or the 3D camera limitations. Moreover, the performance of these methods is hard to maintain satisfactory results when they are directly applied to 2D body images. To tackle these problems, we propose to estimate accurate BMI results from 2D body images by an end-to-end Convolutional Neural Network (CNN) with attention guidance. The proposed method is evaluated on our collected dataset. Extensive experiments confirm that the proposed framework outperforms state-of-the-art approaches in most cases.

## Install

Our code is tested with PyTorch 1.4.0, CUDA 10.0 and Python 3.6. It may work with other versions.

You will need to install some python dependencies(either `conda install` or `pip install`)

```
scikit-learn
scipy
tensorboardX
opencv-python
```
### Dataset
You can download the dataset from the [link](https://pan.baidu.com/s/1Pr0Z7UCHG2R1pnP3a2BVkw), the code is `FVL1`.
## Usage
### Training

```
python main.py --set Ours --root $YOU_PATH$ -b 32
```
### Testing

```
python main.py --set Ours --root $YOU_PATH$ --test_mode True --resume $MODEL_PATH$
```
