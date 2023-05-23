library(tidyverse) 
library (corrplot)
library(caret)
library(rpart)

df <- read.csv("E:/FIFA 20/FIFA20.csv")

#可以看到非常的多，我们只取可能有用的部分，诸如"player_url"一类数据需要删掉。
#head (df)
#names (df) 
#str (df) 


#按“去除所有空缺数据”处理球员数据，相比“赋予平均值”效果更好，接下来会证明 
data <- df[,c('height_cm','weight_kg','overall','potential','value_eur','release_clause_eur','wage_eur','skill_moves','pace','shooting','passing','dribbling','defending','physic')] 
data <- data[!is.na(data$passing),]  #守门员没有passing数据，精准打击
data <- data[!is.na(data$release_clause_eur),]
corr_matrix <- cor(data)
corrplot(corr_matrix, method = "square",diag = F, type = "lower")

#门将是特殊部分，需要单独处理
data_gk <- df[,c('height_cm','weight_kg','overall','potential','value_eur','release_clause_eur','wage_eur','gk_diving','gk_handling','gk_kicking','gk_reflexes','gk_speed','gk_positioning')] 
data_gk <- data_gk[!is.na(data_gk$gk_diving),]
data_gk <- data_gk[!('value_eur' == 0),] #自由球员身价是0，无参考价值,应该去掉
data_gk <- data_gk[!is.na(data_gk$release_clause_eur),]
corr_matrix_gk <- cor(data_gk)
corrplot(corr_matrix_gk, method = "square",diag = F, type = "lower")



###########################对非门将球员用线性回归模型处理############################
#分训练集和实验集
set.seed(313)
ind <- sample(2,nrow(data),replace=T,prob=c(0.7,0.3))
data_train <- data[ind == 1,]
data_test <- data[ind == 2,]
#建模
data_model <- lm(value_eur~ height_cm + weight_kg + overall + release_clause_eur + potential + wage_eur + skill_moves + pace + shooting + passing + dribbling + defending + physic,data_train)
#获取误差
data_pred = predict(data_model,data_test)
data_pred <- ifelse(data_pred<0,0,data_pred)
data_actual <- data_test$value_eur
data_error <- data_actual - data_pred

#这个图显然范围太大了
ggplot(data.frame(data_error), aes(x = data_error)) +
  geom_histogram(binwidth = 50000, color = "black", fill = "white") +
  xlab("Error") +
  ylab("Count")

#确定范围为+-1500000，牺牲了304 个极端数据
ggplot(data.frame(data_error), aes(x = data_error, fill = (data_error > 0))) +
  geom_histogram(binwidth = 50000, color = "black") +
  scale_x_continuous(limits = c(-1500000, 1500000)) +
  scale_fill_manual(values = c("red", "green")) +  # 手动指定颜色
  xlab("Error") + ylab("Count") 

# 绘制残差图，可惜尾部误差普遍较高。
plot(data_pred, data_error, xlab = "Predicted Value", ylab = "Residuals", xlim = c(0, 1000000),ylim = c(-3000000, 3000000))
abline(h = 0, col = "red")

#检验
#创建一个交叉验证控制对象ctrl，指定交叉验证方法为"cv"，即k折交叉验证，k的值为10。
ctrl <- trainControl(method = "cv", number = 10)
# 进行交叉验证
data_model_cv1 <- train(value_eur ~., data = data_train, method = "lm",trControl = ctrl)
# 查看交叉验证结果
data_model_cv1$results

#################################################################################################

#补充内容：用平均值填补/去除所有空缺数据 可视化对比

#按“用平均值填补”处理球员数据
data <- df[,c('height_cm','weight_kg','overall','potential','value_eur','release_clause_eur','wage_eur','skill_moves','pace','shooting','passing','dribbling','defending','physic')] 
data <- data[!is.na(data$passing),]  #守门员没有passing数据，精准打击
data <- data[data$value_eur != 0,]#自由球员身价是0，无参考价值，应该去掉
data$release_clause_eur[is.na(data$release_clause_eur)] <- mean(data$release_clause_eur, na.rm = TRUE)


#分训练集和实验集
set.seed(313)
ind <- sample(2,nrow(data),replace=T,prob=c(0.7,0.3))
data_train <- data[ind == 1,]
data_test <- data[ind == 2,]
#建模
data_model <- lm(value_eur~ height_cm + weight_kg + overall + release_clause_eur + potential + wage_eur + skill_moves + pace + shooting + passing + dribbling + defending + physic,data_train)
#获取误差
data_pred = predict(data_model,data_test)
data_pred <- ifelse(data_pred<0,0,data_pred)
data_actual <- data_test$value_eur
data_error <- data_actual - data_pred

#检验
#创建一个交叉验证控制对象ctrl，指定交叉验证方法为"cv"，即k折交叉验证，k的值为10。
ctrl <- trainControl(method = "cv", number = 10)
# 进行交叉验证
data_model_cv2 <- train(value_eur ~., data = data_train, method = "lm",trControl = ctrl)
# 查看交叉验证结果
data_model_cv2$results


# 将评估结果单独列出
df1 <- as.data.frame(data_model_cv1$results)
df2 <- as.data.frame(data_model_cv2$results)

# 合并RMSE/MAE
df <- data.frame(Model = rep(c("Blank", "Average"), each = 2),
                 Metric = rep(c("RMSE", "MAE"), 2),
                 Value = c(df1$RMSE,df1$MAE, mean(df2$RMSE),mean(df2$MAE)))
# 绘制关于RMSE/MAE的柱状图
ggplot(df, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Comparison of Two Models",
       x = "Metric", y = "Value", fill = "Model")

# 合并数据框，R方
df <- data.frame(Model = rep(c("Blank", "Average"), each = 1),
                 Metric = rep(c('Rsquared'), 2),
                 Value = c(df1$Rsquared,mean(df2$Rsquared)))
# 绘制关于R方的柱状图
ggplot(df, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Comparison of Rsquared Two Models",
       x = "Metric", y = "Value", fill = "Model")


#################################对非门将球员用决策树模型处理####################################
# 划分训练集和测试集
set.seed(313)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
data_train <- data[ind == 1, ]
data_test <- data[ind == 2, ]

# 构建决策树模型
data_model <- rpart(value_eur ~ height_cm + weight_kg + overall + release_clause_eur + potential + wage_eur + skill_moves + pace + shooting + passing + dribbling + defending + physic, data = data_train)

# 对测试集进行预测
data_pred = predict(data_model,data_test)
data_pred <- ifelse(data_pred<0,0,data_pred)


# 计算预测误差
data_actual <- data_test$value_eur
data_error <- data_actual - data_pred


# 绘制误差直方图，限制误差范围为±1000000
ggplot(data.frame(data_error), aes(x = data_error, fill = (data_error > 0))) +
  geom_histogram(binwidth = 50000, color = "black") +
  scale_x_continuous(limits = c(-1000000, 1000000)) +
  scale_fill_manual(values = c("red", "green")) +
  xlab("Error") +
  ylab("Count") 


# 使用交叉验证进行模型选择和优化
ctrl <- trainControl(method = "cv", number = 10)
data_model_cv2 <- train(value_eur ~ ., data = data_train, method = "rpart", trControl = ctrl)
# 查看交叉验证结果，我关心的数据主要是RMSE：均方根误差，MAE：平均绝对误差，Linear Regression：R方。
data_model_cv2$results



# 将评估结果转换为数据框
df1 <- as.data.frame(data_model_cv1$results)
df2 <- as.data.frame(data_model_cv2$results)

# 合并数据框，RMSE/MAE
df <- data.frame(Model = rep(c("Linear Regression", "Decision Tree"), each = 2),
                 Metric = rep(c("RMSE", "MAE"), 2),
                 Value = c(df1$RMSE,df1$MAE, mean(df2$RMSE),mean(df2$MAE)))
# 绘制关于RMSE/MAE的柱状图
ggplot(df, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Comparison of Evaluation Metrics between Two Models",
       x = "Metric", y = "Value", fill = "Model")

# 合并数据框，R方
df <- data.frame(Model = rep(c("Linear Regression", "Decision Tree"), each = 1),
                 Metric = rep(c('Rsquared'), 2),
                 Value = c(df1$Rsquared,mean(df2$Rsquared)))
# 绘制关于R方的柱状图
ggplot(df, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Comparison of Rsquared Two Models",
       x = "Metric", y = "Value", fill = "Model")
