# General_Image_Classify_Model

* Purpose
  这份代码是为了复用而创建的，因为我发现如果用keras的这个框架去运用imagenet上预训练好的模型是很简单的，并且重复代码多。另外设置成这个目录结构就可以在各种不同的图像分类比赛中作出baseline。
  
* 目录结构
  * code : 存放我用jupyter测试后能用的代码，功能从文件名就可以看出来（gap_train.py和pretrain.py是例外这两个是一套的，先执行pretrain会获得预训练好的特征，然后在gap_train里通过融合。这是[使用非常少的数据构建强大的图像分类模型](https://github.com/ictar/python-doc/blob/master/Machine%20Learning/%E4%BD%BF%E7%94%A8%E9%9D%9E%E5%B8%B8%E5%B0%91%E7%9A%84%E6%95%B0%E6%8D%AE%E6%9E%84%E5%BB%BA%E5%BC%BA%E5%A4%A7%E7%9A%84%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B.md)里的代码稍加修改，具体内容可看这篇）
  * data : 这里面的目录结构应该严格按照keras的flow_from_directory规则(值得一提的是，如果是猪脸识别比赛的数据，你需要将数据shuffle然后得到两个不同的文件夹包含训练集和验证集，去执行code里的split_into_train_and_val.py就行)
  * result : 存放结果数据
  * model : 存放模型参数
  * *.ipynb : 这些是我做测试的jupyter文件，没有删，因为有些代码还是能复用。
  
* 代码说明（由于这是一份基于猪脸识别比赛做的简单代码，所以里面分类是30类，如果要用于别的更多的类别需自行改动）
  * flood_algorithm.py : 对图片使用满水算法
  * inception_V4_train.py : 利用inceptionRestnetV2训练,并且有fine-tune,可在代码里修改参数
  * load_and_train_V4.py : 利用之前训练过的模型参数继续训练，修改代码里的参数就可以用于不同的模型
  * inceptionV3_train.py : 利用inceptionV3训练,并且有fine-tune,可在代码里修改参数
  * load_model_and_getResult.py : 加载模型并计算得到结果并保存
  * gap_train.py 和 pretrain.py : 前面讲过
  * split_into_train_and_val.py : 前面讲过
  
  
  
**important:这份代码最终只是为了方便我实现一个baseline，做比赛还是需要从头构建模型**
