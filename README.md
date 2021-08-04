# Learning Facial Representations from the Cycle-consistency of Face (ICCV 2021)

This repository contains the code for our ICCV2021 paper by Jia-Ren Chang, Yong-Sheng Chen, and Wei-Chen Chiu.

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
| [TCAE](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Self-Supervised_Representation_Learning_From_Videos_for_Facial_Action_Unit_Detection_CVPR_2019_paper.pdf) | 2.87 % | 2.61 % |
| [BMVC’20](https://www.bmvc2020-conference.com/assets/papers/0861.pdf) | 3.89 % | 3.33 % |

## Usage

### Download Pretrained Model

### Test translation

## Contacts
followwar@gmail.com

Any discussions or concerns are welcomed!
