import os
import paddle
import numpy as np


class bridger(object):
    def __init__(self, model, optimizer, loss_fn, metric):
        self.model = model          # 模型
        self.optimizer = optimizer  # 优化器
        self.loss_fn = loss_fn      # 损失函数
        self.metric = metric        # 评价指标

    # 模型训练
    def train(self, dataset, reg_lambda, model_dir):
        X, y = dataset
        self.optimizer(self.model, X, y, reg_lambda)

        # 保存模型
        self.save_model(model_dir)

    # 模型评价
    def evaluate(self, dataset, **kwargs):
        X, y = dataset
        y_pred = self.model(X)
        result = self.metric(y_pred, y)

        return result

    # 模型预测
    def predict(self, X, **kwargs):
        return self.model(X)

    # 模型保存
    def save_model(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        params_saved_path = os.path.join(model_dir, 'params.pdtensor')
        paddle.save(self.model.params, params_saved_path)

    # 模型加载
    def load_model(self, model_dir):
        params_saved_path = os.path.join(model_dir,'params.pdtensor')
        self.model.params = paddle.load(params_saved_path)
