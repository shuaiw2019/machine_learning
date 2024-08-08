import os
import paddle
import numpy as np


class Runner(object):
    def __init__(self, model, optimizer, loss_fn, metric):
        self.model = model          # 模型
        self.optimizer = optimizer  # 优化器
        self.loss_fn = loss_fn      # 损失函数
        self.metric = metric        # 评价指标
        self.train_scores = []      # 记录训练过程中评价指标变化情况
        self.dev_scores = []
        self.train_loss = []        # 记录训练过程中损失函数变化情况
        self.dev_loss = []

    # 模型训练
    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get('num_epochs', 0)       # 传入训练回合数，如果没有则默认为0
        log_epochs = kwargs.get('log_epochs', 100)     # 传入log打印频率，如果没有则默认为100
        save_path = kwargs.get('save_path', '../Bridge/best_model.pdparams')    # 传入模型保存路径
        print_grads = kwargs.get('print_grads', None)     # 传入梯度打印函数，如果没有则默认为None
        best_score = 0        # 记录全局最优指标
        # 进行训练
        for epoch in range(num_epochs):
            X, y = train_set
            # 获取模型预测
            logits = self.model(X)
            # 计算交叉熵损失
            trn_loss = self.loss_fn(logits, y).item()
            self.train_loss.append(trn_loss)
            # 计算评价指标
            trn_score = self.metric(logits, y).item()
            self.train_scores.append(trn_score)
            # 计算参数梯度
            self.model.backward(y)
            if print_grads is not None:
                # 打印每一层的梯度
                print_grads(self.model)
            # 更新模型参数
            self.optimizer.step()
            dev_score, dev_loss = self.evaluate(dev_set)
            # 如果当前指标为最优指标，保存改模型
            if dev_score > best_score:
                self.save_model(save_path)
                print(f'best accuracy performance has been updated:, {best_score:.5f} --->  {dev_score:.5f}')
                best_score = dev_score
            if epoch % log_epochs == 0:
                print(f'[Train] epoch: {epoch}, loss: {trn_loss}, score: {trn_score}')
                print(f'[Dev] epoch: {epoch}, loss: {dev_loss}, score: {dev_score}')

    # 模型评价
    def evaluate(self, dataset, **kwargs):
        X, y = dataset
        logits = self.model(X)         # 计算模型输出
        loss = self.loss_fn(logits, y).item()
        self.dev_loss.append(loss)
        # 计算评价指标
        score = self.metric(logits, y).item()
        self.dev_scores.append(score)

        return score, loss

    # 模型预测
    def predict(self, X, **kwargs):

        return self.model(X)

    # 模型保存
    def save_model(self, save_path):

        paddle.save(self.model.params, save_path)

    # 模型加载
    def load_model(self, model_path):
        self.model.params = paddle.load(model_path)
