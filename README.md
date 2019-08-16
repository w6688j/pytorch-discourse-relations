# pytorch-discourse-relations

#### 介绍
基于Pytorch-0.4.1，Python-3.7的多种篇章关系识别模型  
RNN, RNNATT17, GRN16, KEANN, KEANN-KG

#### 环境要求
PyTorch 0.4.1  
Python 3.7  
Numpy 1.15.4  
Scikit-learn 0.20.2  
Pandas 0.24.1

#### 训练
python train_use_conf.py --conf 'conf/模型配置文件名称.conf'  
rnnatt17_glove_300v : python train_use_conf.py --conf 'conf/rnnatt17_glove_300v.conf'  
  

#### 测试
python test_use_conf.py --conf 'conf/模型配置文件名称.conf'  
rnnatt17_glove_300v : python test_use_conf.py --conf 'conf/rnnatt17_glove_300v.conf'  

 运行tensorboardX  
 tensorboard --logdir=runs