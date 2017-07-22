![](https://raw.githubusercontent.com/NELSONZHAO/zhihu/master/cifar_cnn/example_pic.png)


# 说明
该部分包含两个文件：

- CIFAR_KNN.ipynb: 用K近邻实现的图像分类
- CIFAR_CNN.ipynb: 用CNN实现的图像分类文件

# 版本
```Python3```和```TensorFlow 1.0```

# 其他说明
文件中加载CIFAR数据是通过本地加载的，如果需要使用floyd线上环境跑代码的同学，可以直接使用floyd上的CIFAR数据，而不需要push上去。只需要将```CIFAR_CNN.ipynb```对应代码改为：

	import tarfile
	cifar10_path = 'cifar-10-batches-py'
	tar_gz_path = '/input/cifar-10/python.tar.gz' # 这是floyd上存储CIFAR数据的地址
	with tarfile.open(tar_gz_path) as tar:
	    tar.extractall()
	    tar.close()
	    
在使用floyd命令上传代码文件时指定数据编号：

	floyd run --gpu --env tensorflow-1.0 --mode jupyter --data diSgciLH4WA7HpcHNasP9j