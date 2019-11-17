train.csv: 训练集，每个月前20天每个小时的气象资料（每小时有18种测资）共12个月

test.csv: 测试集，排除train.csv剩余的资料 取连续9小时的资料当作feature 预测第10小时的PM2.5值 总共240笔

ans.csv: 测试集数据对应的label

pm2.5_prediction.ipynb 与 pm2_5_prediction.py 为参考的练习答案
