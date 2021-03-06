# Learning Facial Representations from the Cycle-consistency of Face (ICCV 2021)

This repository contains the code for our ICCV2021 paper by Jia-Ren Chang, Yong-Sheng Chen, and Wei-Chen Chiu.

[Paper Arxiv Link](https://arxiv.org/pdf/2108.03427.pdf)

## Contents

1. [Introduction](#introduction)
2. [Results](#results)
3. [Usage](#usage)
4. [Contacts](#contacts)

## Introduction
In this work, we introduce cycle-consistency in facial characteristics as free supervisory signal to learn facial representations from unlabeled facial images. The learning is realized by superimposing the facial motion cycle-consistency and identity cycle-consistency constraints. The main idea of the facial motion cycle-consistency is that, given a face with expression, we can perform de-expression to a neutral face via the removal of facial motion and further perform re-expression to reconstruct back to the original face. The main idea of the identity cycle-consistency is to exploit both de-identity into mean face by depriving the given neutral face of its identity via feature re-normalization and re-identity into neutral face by adding the personal attributes to the mean face.

<img align="center" src="https://user-images.githubusercontent.com/11732099/128152906-4ebc6711-7fc0-431a-9145-4b2c7f12a7fb.png">

## Results

#### More visualization

<img align="center" src="https://user-images.githubusercontent.com/11732099/128154030-7936207d-a8f2-4a57-80e2-f5565515de00.png">

#### Emotion recognition

We use linear protocol to evaluate learnt representations for emotion recognition. We report accuracy (%) for two dataset.

| Method | FER-2013 | RAF-DB |
|---|---|---|
| Ours | 48.76 % | 71.01 % |
| [FAb-Net](https://arxiv.org/abs/1808.06882) | 46.98 % | 66.72 % |
| [TCAE](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Self-Supervised_Representation_Learning_From_Videos_for_Facial_Action_Unit_Detection_CVPR_2019_paper.pdf) | 45.05 % | 65.32 % |
| [BMVC’20](https://www.bmvc2020-conference.com/assets/papers/0861.pdf) | 47.61 % | 58.86 % |

#### Head  pose  regression

We use linear regression to evaluate learnt representations for head pose regression.

| Method | Yaw | Pitch | Roll |
|---|---|---|---|
| Ours | 11.70 | 12.76 | 12.94 |
| [FAb-Net](https://arxiv.org/abs/1808.06882) | 13.92 | 13.25 | 14.51 |
| [TCAE](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Self-Supervised_Representation_Learning_From_Videos_for_Facial_Action_Unit_Detection_CVPR_2019_paper.pdf) | 21.75 | 14.57 | 14.83 |
| [BMVC’20](https://www.bmvc2020-conference.com/assets/papers/0861.pdf) | 22.06 | 13.50 | 15.14 |


#### Person recognition

We directly adopt learnt representation for person recognition.

| Method | LFW | CPLFW |
|---|---|---|
| Ours | 73.72 % | 58.52 % |
| [VGG-like](https://arxiv.org/abs/1803.01260) | 71.48 % | - |
| LBP | 56.90 % | 51.50 % |
| HoG | 62.73 % | 51.73 % |

#### Frontalization

The frontalization results from LFW dataset. 

<img align="center" src="https://user-images.githubusercontent.com/11732099/128305185-0020f0b8-7a90-4394-b71d-35a95a05bec2.png">


#### Image-to-image Translation

The image-to-image translation results. 

<img align="center" src="https://user-images.githubusercontent.com/11732099/128305387-fbd9c7c6-6431-43c1-9109-9e766d065cda.png">


## Usage

### From Others

Thanks to all the authors of these awesome repositories.
[SSIM](https://github.com/Po-Hsun-Su/pytorch-ssim)
[Optical Flow Visualization](https://github.com/tomrunia/OpticalFlow_Visualization)

### Download Pretrained Model

[Google Drive](https://drive.google.com/file/d/1dDuXLyn3AFclGos-Ku2geMBWE2v4a2Y9/view?usp=sharing)

### Test translation

```
python test_translation.py --loadmodel (pretrained model) \
```

and you can get like below

<img align="center" src="https://github.com/JiaRenChang/FaceCycle/blob/master/Test_translation/translation0.png">

### Replicate RAF-DB results

Download pretrained model and [RAF-DB](http://www.whdeng.cn/RAF/model1.html)

```
python RAF_classify.py --loadmodel (pretrained model) \
                       --datapath (your RAF dataset path) \
                       --savemodel (your path for saving)
```

You can get 70~71% accuracy with basic emotion classification (7 categories) using linear protocol.

## Contacts
followwar@gmail.com

Any discussions or concerns are welcomed!
