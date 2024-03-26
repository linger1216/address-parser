# service
bentoml serve service.py:svc --reload

# 打包
bentoml build --debug -f bentofile.yaml


docker run -p 3000:3000 parse:afbmmdxlewahh5xo

## todo
1. 增加配置
2. test eval train的作用丰富, 最好模型的选择
3. 增加http接口
4. 把规则引擎, 知识库, 创造预料都给集成寄来
5. 优化文件夹结构