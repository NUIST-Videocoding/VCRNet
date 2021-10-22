# VCRNet
Guided by the free-energy principle, generative adversarial networks (GAN)-based no-reference image quality assessment (NR-IQA) methods have improved the image quality
prediction accuracy. However, the GAN cannot well handle the restoration task for the free-energy principle-guided NR-IQA
methods, especially for the severely destroyed images, which results in that the quality degradation relationship between the
distorted image and its restored image cannot be accurately built.To address this problem, a visual compensation restoration network (VCRNet)-based NR-IQA method is proposed, which
uses a non-adversarial model to efficiently handle the distorted image restoration task. The proposed VCRNet consists of a
visual restoration network and a quality estimation network.
![./image-20211022140814450](https://github.com/NUIST-Videocoding/VCRNet/blob/main/VCRNet/image-20211022140814450.png)

### Dataset
| Dataset   | Links                                                       |
| --------- | ----------------------------------------------------------- |
| LIVE      | https://live.ece.utexas.edu/research/quality/index.htm      |
| TID2013   | http://r0k.us/graphics/kodak/                               |
| KONIQ-10K | http://database.mmsp-kn.de/koniq-10k-database.html          |
| CSIQ      | https://pan.baidu.com/s/1XCSafnf3SlbgePJuMq5M5w  pass: w7dh |

### Training and Testing
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --dataset live
```

### Note
- The pretrained EfficientNet-B0 is installed by:
```bash
pip install efficientnet_pytorch
```
- During training process, the dataset is split into "training_set" and "testing_set", and the testing process are also been performed, and the testing results can be seen in TEST_SRCC and TEST_PLCC

## Requirements
- PyTorch=1.1.0
- Torchvision=0.3.0
- numpy=1.16.0
- scipy=1.2.1
- argparse=1.4.1
- h5py=2.10.0
- efficientnet_pytorch=0.7.1