saliencyDectection
==================

this is the implementation of salient detection algorithm of CHENG Mingming


saliencyDectection Using histogram based method 
------------------
this is implemented in saliencyDetectHistogram.m

saliencyDectection Using region based method
------------------ 
this is implemented in saliencyDetectRegion.m

testing on benchmark
-------------------
I use the testRC.m and testHC.m two files to test the methods on bench mark on testset folder, to save space, I do not upload the testset folder, the smallset is only a sample of the large set

test Result
-------------------
data file are in the result folder

reference
------------------
the original paper is in 
@conference{11cvpr/Cheng_Saliency,
title={Global Contrast based Salient Region Detection},
author={Ming-Ming Cheng and Guo-Xin Zhang and Niloy J. Mitra and Xiaolei Huang and Shi-Min Hu},
booktitle={IEEE CVPR},
pages={409--416},
year={2011},
}