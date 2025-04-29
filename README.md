数据集下载链接： https://pan.baidu.com/s/144-ZTWV60bzoQUGNFLuUQw?pwd=2025 提取码: 2025 
这个数据集是华为云人工智能大赛2020年的垃圾分类比赛的数据集，比赛官网：https://competition.huaweicloud.com/information/1000007620/circumstances?zhishi=
这个模型是做40分类的任务，具体的垃圾名字在文件：mapping.txt中。我感觉我这个项目也算一个分类吧。因为写的太垃圾了。哈哈哈哈哈哈哈~


写了几个模形：
其中我自己用来练手的是：CNN_MultiScale 简单的多尺度学习，好垃圾。过拟合严重测试。准确率只有34%。。。
还有几个预训练模型是：ResNet18 和 ResNet34 和 ResNet50 数字越大，网络参数越大。
微调了一下这几个模型：我这里用的是ResNet34跑了一个82的准确率（目前最好）
具体是怎么微调的？：原来的ResNet用来训练数据集，发现训练几个epoch就导致模型过拟合了，所以在模型的全连接层之前，加了Dropout和正则化。训练过程中使用早停机制，发现几个epoch就收敛了，然后直接停止了。不得不说ResNet牛逼。


下载好数据集，解压到项目的根目录即可。
终端安装：pip install -r requirements.txt  安装所需要的环境
如果要重新训练：直接运行train.py 文件。我里面设置了30个epoch，但是加了早停机制，其实差不多5个epoch就停止了。
如果要直接装逼：直接运行app.py 文件。上传一张垃圾图片就能识别出来。
一天帮同学写的毕设，大家随便看看吧。
