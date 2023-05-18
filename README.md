# Smart Word Suggestions for Writing Assistance

This repository contains the code and pre-trained models for our paper [Smart Word Suggestions for Writing Assistance](https://arxiv.org/pdf/2305.09975.pdf).

## Overview

This paper introduces "Smart Word Suggestions" (SWS) task and benchmark. 
Unlike other works, SWS emphasizes end-to-end evaluation and presents a more realistic writing assistance scenario. 
This task involves identifying words or phrases that require improvement and providing substitution suggestions. 
The benchmark includes human-labeled data for testing, a large distantly supervised dataset for training, and the framework for evaluation. 
The test data includes 1,000 sentences written by English learners, accompanied by over 16,000 substitution suggestions annotated by 10 native speakers. 
The training dataset comprises over 3.7 million sentences and 12.7 million suggestions generated through rules. 

![](https://tva1.sinaimg.cn/large/008vOhrAly1hdvfxw0o8fj30u00y2tgd.jpg)

For example, as illustrated in the figure above, the word “inti-mate” in the first sentence should be replaced with
“close”, as “intimate” is not suitable for describing relationships between colleagues.

## Getting Started

The data folder contains all the data, the code folder contains all the baselines. There are corresponding READMEs in the folder. 

## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email Chenshuo (iven@ivenwang.com) and Shaoguang (shaoguang.mao@microsoft.com). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use SWS in your work:

```

```
