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

#### 实验结果
============================= Bi-LSTM =============================  
rnn_no_glove_100v :  
 * Acc@ 52.051  
 * Comparison: Precision@ 0.2229  Recall@ 0.0246  F1@ 0.0430   
 * Contingency: Precision@ 0.4435  Recall@ 0.3943  F1@ 0.4156   
 * Temporal: Precision@ 0.3125  Recall@ 0.0534  F1@ 0.0906   
 * Expansion: Precision@ 0.5548  Recall@ 0.7918  F1@ 0.6515

rnn_glove_300v :   
 * Acc@ 52.637  
 * Comparison: Precision@ 0.2187  Recall@ 0.0451  F1@ 0.0728   
 * Contingency: Precision@ 0.4415  Recall@ 0.4301  F1@ 0.4315   
 * Temporal: Precision@ 0.4375  Recall@ 0.0790  F1@ 0.1277   
 * Expansion: Precision@ 0.5656  Recall@ 0.7786  F1@ 0.6528

rnn_no_glove_300v :  
 * Acc@ 50.684  
 * Comparison: Precision@ 0.2731  Recall@ 0.1988  F1@ 0.2239   
 * Contingency: Precision@ 0.4506  Recall@ 0.4109  F1@ 0.4277   
 * Temporal: Precision@ 0.1723  Recall@ 0.1013  F1@ 0.1235   
 * Expansion: Precision@ 0.5939  Recall@ 0.7042  F1@ 0.6418

============================= RNNATT17 =============================  
rnnatt17_glove_300v(Four classification) : Prec@1 49.219  
rnnatt17_no_glove_300v(Four classification) : Prec@1 46.875
  
============================= GRN16 =============================  
grn16_glove_50v(Four classification) :  
 * Acc@ 51.855  
 * Comparison: Precision@ 0.0000  Recall@ 0.0000  F1@ 0.0000   
 * Contingency: Precision@ 0.0000  Recall@ 0.0000  F1@ 0.0000   
 * Temporal: Precision@ 0.0000  Recall@ 0.0000  F1@ 0.0000   
 * Expansion: Precision@ 0.5186  Recall@ 1.0000  F1@ 0.6786  
  
grn16_no_glove_50v(Four classification) :
 * Acc@ 51.855  
 * Comparison: Precision@ 0.0000  Recall@ 0.0000  F1@ 0.0000   
 * Contingency: Precision@ 0.0000  Recall@ 0.0000  F1@ 0.0000   
 * Temporal: Precision@ 0.0000  Recall@ 0.0000  F1@ 0.0000   
 * Expansion: Precision@ 0.5186  Recall@ 1.0000  F1@ 0.6786 
  
grn16_glove_50v(Binary classification) :   
grn16_no_glove_50v(Binary classification Comparison) : Prec@1 85.547  
  
  
============================= KEANN =============================  
keann_glove_300v(Four classification, learning-rate 0.005 batch-size 128) : Prec@1 52.148  
keann_no_glove_300v(Four classification, learning-rate 0.005 batch-size 128) : Prec@1 51.855    

keann_glove_300v(Four classification, learning-rate 0.001 batch-size 32) :   
 * Acc@ 52.637  
 * Comparison: Precision@ 0.3109  Recall@ 0.2558  F1@ 0.2549   
 * Contingency: Precision@ 0.4675  Recall@ 0.4977  F1@ 0.4628   
 * Temporal: Precision@ 0.2240  Recall@ 0.0963  F1@ 0.1304   
 * Expansion: Precision@ 0.5999  Recall@ 0.6866  F1@ 0.6319 

keann_no_glove_300v(Four classification, learning-rate 0.001 batch-size 32) :  
 * Acc@ 50.684  
 * Comparison: Precision@ 0.2709  Recall@ 0.2520  F1@ 0.2420   
 * Contingency: Precision@ 0.4755  Recall@ 0.3957  F1@ 0.4118   
 * Temporal: Precision@ 0.1833  Recall@ 0.1443  F1@ 0.1523   
 * Expansion: Precision@ 0.5897  Recall@ 0.6807  F1@ 0.6238  
   
============================= KEANN-KG =============================  
keann_kg_glove_50v(Four classification 10 Epoch) :
 * Acc@ 53.711  
 * Comparison: Precision@ 0.1562  Recall@ 0.0553  F1@ 0.0760   
 * Contingency: Precision@ 0.4962  Recall@ 0.4012  F1@ 0.4307   
 * Temporal: Precision@ 0.1875  Recall@ 0.0870  F1@ 0.1126   
 * Expansion: Precision@ 0.5811  Recall@ 0.8011  F1@ 0.6662   
 
 
 运行tensorboardX  
 tensorboard --logdir=runs