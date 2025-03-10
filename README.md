<!-- Title -->
## INA-RSDeblur: Implicit Neural Attention for Removing Blur in Remote Sensing Image

![image text](https://github.com/yaowli468/INA-RSDeblur/blob/main/IMG/Framework.png)

 
## Dependencies
* Linux(Tested on Ubuntu 18.04) 
* Python 3.8.10 (Recomend to use [Anaconda](https://www.anaconda.com/products/individual#linux))
* Pytorch 2.1
* yaml==0.25
* numpy==1.21.2
* pillow==8.3.2 

## Get Started

### Download
* Pretrained model can be downloaded from [HERE](https://pan.baidu.com/s/1KNWp0jc2XlO2pUMVI7ra7Q)(c7ui), please put them to './save/_train_edsr-baseline-liif/'

### Testing
1. Run the following commands to test.
    ```sh
    python test.py --config ./configs/test/test-blur.yaml --model ./save/_train_edsr-baseline-liif/epoch-best.pt --gpu 0
    ```

### Training
1. Run the following command to prepare dataset.
   ```sh  
   python train_liif.py --config configs/train-div2k/train_edsr-baseline-liif.yaml
   ```

## Acknowledgments
This code is based on [liif](https://github.com/yinboc/liif). Thanks for their greate works.

 



