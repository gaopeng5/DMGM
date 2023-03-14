# DMGM
Code and dataset for ICRA2023 paper "Deep Masked Graph Matching for Correspondence Identification in Collaborative Perception"

## Overview

The overview of our approach is shown as follows:

<p align="center">
<img src="https://user-images.githubusercontent.com/58457277/224859225-d0ed29f8-263b-4ca6-afb4-b55791c3e450.png" width="600" height="400"/>
<p >

## Dataset
![fig3-1](https://user-images.githubusercontent.com/58457277/224859588-4e8bbe0a-1249-4dfa-b62f-56810526e30e.png )

1. To test our approach, we upload 1000 processed data instances, which can be found in testdata/data folder.

2. Our processed test dataset can be found [here](https://drive.google.com/file/d/1-3J5Oic8fo3fttWTF-s3pwe6_7xVFxHD/view?usp=sharing)

3. Our full raw dataset (one scenario) used in this paper can be found [here]

4. Our extended dataset (including four scenarios) can be found [here](https://drive.google.com/drive/u/1/folders/1_OmWAn2dGzXk0-37Lk0Il5e8r9FsOG53)

## Requirements

We recommend python3.9. You can install required dependencies by:
    
---
    pip -r install requirements.txt
---

## Test

We provide pre-trained model in checkpoint folder. To reproduce our results shown in the paper, please run:

---
    python run test_rural_ours.py
---


## Citation
If you use DMGM in a scientific publication, we would appreciate using the following citation:

---
    @article{Deep Masked Graph Matching for Correspondence Identification in Collaborative Perception,
        author = {Peng Gao, Qingzhao Zhu, Hongsheng Lu, Chuang Gan, and Hao Zhang},
        journal = {ICRA 2023},
        year = {2023}
---
