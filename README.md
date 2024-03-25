input/ ：该⽂件夹包含机器学习项⽬的所有输⼊⽂件和数据。如果您正在开发 NLP 项⽬，您可以
将 embeddings 放在这⾥。如果是图像项⽬，所有图像都放在该⽂件夹下的⼦⽂件夹中。

src/ ：我们将在这⾥保存与项⽬相关的所有 python 脚本。如果我说的是⼀个 python 脚本，即任
何\_.py ⽂件，它都存储在 src ⽂件夹中。

models/ ：该⽂件夹保存所有训练过的模型。

notebook/ ：所有 jupyter notebook（即任何\_.ipynb ⽂件）都存储在笔记本 ⽂件夹中。
README.md ：这是⼀个标记符⽂件，您可以在其中描述您的项⽬，并写明如何训练模型或在⽣
产环境中使⽤。

LICENSE ：这是⼀个简单的⽂本⽂件，包含项⽬的许可证，如 MIT、Apache 等。关于许可证的
详细介绍超出了本书的范围。




# service

bentoml serve service.py:svc --reload

# 打包
bentoml build --debug -f bentofile.yaml


## todo
1. 增加配置
2. test eval train的作用丰富, 最好模型的选择
3. 增加http接口
4. 把规则引擎, 知识库, 创造预料都给集成寄来
5. 优化文件夹结构