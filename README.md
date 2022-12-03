# MoADNet
MoADNet: Mobile Asymmetric Dual-Stream Networks for Real-Time and Lightweight RGB-D Salient Object Detection

This is the official implementation of "MoADNet: Mobile Asymmetric Dual-Stream Networks for Real-Time and Lightweight RGB-D Salient Object Detection" as well as the follow-ups. The paper has been published by IEEE Transactions on Circuits and Systems for Video Technology, 2022. The paper link is [here](https://ieeexplore.ieee.org/document/9789193).
****

## Content
* [Run MoADNet code](#Run)
* [Saliency maps](#Saliency)
* [Evaluation tools](#Evaluation)
* [Citation](#Citation)
****

## Run MoADNet code
- Train <br>
  run `python train.py` <br>
  \# set '--train-root' to your training dataset folder
  
- Test <br>
  run `python test.py` <br>
  \# set '--test-root' to your test dataset folder <br>
  \# set '--ckpt' as the correct checkpoint <br>
****

## Saliency maps
  - The saliency maps can be approached in [Baidu Cloud](https://pan.baidu.com/s/1SXAC1DtgeuyQ_WxlyI9VeQ) (fetach code is moad). Note that the results provided in paper are the average values after several training times.
****

## Evaluation tools
- The evaluation tools, training and test datasets can be obtained in [RGBD-SOD-tools](https://github.com/kingkung2016/RGBD-SOD-tools).
****

## Citation
```
@ARTICLE{jin2022moadnet,
  author={Jin, Xiao and Yi, Kang and Xu, Jing},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={MoADNet: Mobile Asymmetric Dual-Stream Networks for Real-Time and Lightweight RGB-D Salient Object Detection}, 
  year={2022},
  volume={32},
  number={11},
  pages={7632-7645}
}
```
****


