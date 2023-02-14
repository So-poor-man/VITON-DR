# VITON-DR: Virtual try-on based on clothing detail retention    
Official code for paper 'VITON-DR: Virtual try-on based on clothing detail retention' (__Unpublished__).    
This code is an improvement proposed on the basis of __ACGPN__, which improves the ___Clothes Warping Module___ in the virtual try on structure proposed by __ACGPN__.

## Installation   
`python 3.8.11`    
`pytorch 1.9,0`    
`numpy 1.21.5`   
`opencv-python 4.6.0`    

## test     
`python test.py`  
__Note that__ the pre-trained models we use are from __ACGPN__ and the test dataset is from __VITON__.

##Checkpoint 
We used the pre-trained models `latest_net_G.pth`, `latest_net_G1.pth`, `latest_net_G2.pth` from ACGPN, which you can download [here](https://drive.google.com/file/d/1UWT6esQIU_d4tUm8cjxDKMhB8joQbrFx/view?usp=sharing) .

## Dataset
**VITON Dataset** This dataset is presented in [VITON](https://github.com/xthan/VITON), containing 19,000 image pairs, each of which includes a front-view woman image and a top clothing image. After removing the invalid image pairs, it yields 16,253 pairs, further splitting into a training set of 14,221 paris and a testing set of 2,032 pairs.
