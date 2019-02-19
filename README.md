# 12306验证码识别

### 基于深度学习的12306验证码识别

目前仅整理了12306验证码图片部分分类代码，关于验证码文字部分的代码尚未整理，不过原理相同。
解压data目录下的dataset.zip，目录结构应该为
* data
* ---dataset
* ------各个分类名
* ---------图片.jpg
<br>
<br>
测试识别准确率为96%，出问题的多在容易混淆的图片分类，例如中国结和剪纸，蜡烛和烛台

### 训练模型
``python train.py``

### 使用模型
``python classify.py``

### 参考连接
https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/