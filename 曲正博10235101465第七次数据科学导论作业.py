import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score 

# 定义自行车数据集的文件路径
file_path = r'C:\Users\ROG\Desktop\bike.csv'  

# 读取CSV文件到DataFrame
df = pd.read_csv(file_path)  

# 移除数据集中的'id'列
df = df.drop(columns=['id']) 

# 仅保留城市代码为1的记录
df = df[df['city'] == 1]

# 移除数据集中的'city'列
df = df.drop(columns=['city'])

# 将'hour'列转换为白天（1）和夜晚（0）
df['hour'] = ((df['hour'] >= 6) & (df['hour'] < 19)).astype(int)

# 提取目标变量'y'列并将其转换为独立的DataFrame
y_column = df.pop('y')

# 将目标变量转换为向量
y = y_column.values.reshape(-1, 1)

# 将特征变量转换为二维数组
X = df.to_numpy()

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# 初始化Min-Max缩放器
min_max_scaler = MinMaxScaler()

# 对训练集和测试集的特征归一化
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

# 对训练集和测试集的目标变量进行归一化
y_train = min_max_scaler.fit_transform(y_train)
y_test = min_max_scaler.transform(y_test)

# 初始化线性回归模型
model = LinearRegression()

# 使用训练集数据拟合模型
model.fit(X_train, y_train)

# 预测测试集的目标变量
y_test_pred = model.predict(X_test)

# 计算测试集的均方误差
mse = mean_squared_error(y_test, y_test_pred) 

# 计算测试集的均方根误差
rmse = np.sqrt(mse)

# 输出测试集的均方根误差
print(f"Testing RMSE: {rmse}")