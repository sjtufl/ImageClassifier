#### 配置文件

- HOG orientations, cell大小, block大小等特征提取参数
- 训练使用的特征数据、测试数据、保存模型文件的路径

` p.s. Cifar-10原始数据直接在extractFeat.py里指定路径`

#### 源代码说明

- extractFeat.py & extractFeat4report.py 提取原始数据的HOG特征并保存到config文件指定的目录中
- analysis4report & classifier.py 实现模型training, validation以及generalization test
- 其他文件主要是画图、加载配置文件等辅助工具