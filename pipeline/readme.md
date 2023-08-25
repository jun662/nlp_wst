# pipleline for gpt2(gpt2流水线并行)
## 传统的模型并行
![传统模型并行](https://camo.githubusercontent.com/9bd0e5bcf663ac4ae992ce3098dd8a7a0bafe99b8d46eae136187c1fba0f5b77/68747470733a2f2f7079746f7263682e6f72672f646f63732f737461626c652f5f696d616765732f6e6f5f706970652e706e67)
## 流水线并行见下图
![流水线并行](https://camo.githubusercontent.com/4314e4ad94a0d2dc3462ea42c4fc175b74db129322b408fdca5332fba40f2804/68747470733a2f2f7079746f7263682e6f72672f646f63732f737461626c652f5f696d616765732f706970652e706e67)

本文代码是基于pytorch提供的[pipeline parallelism](https://pytorch.org/docs/stable/pipeline.html)功能，参考了transformer包提供的gpt2的代码，给gpt2实现了一套流水线并行训练代码。

### 改造注意点
各个layer层可被传输切片的数据必须是可被迭代的张量数据

### 代码构造
* 1 pipegpt2.py为主体代码，可参考进行训练
* 2 structure_data.py为模拟构造的训练参数（本文没有具体的真实数据，是随机构造的数据），可参考对你想要的数据进行改造训练

