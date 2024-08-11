"""
模型评估指标库
"""
import paddle


# 准确率
def accuracy(preds, labels):
    """
    输入：
    -params: preds: 预测值，二分类：shape=[N,1]，多分类：shape=[N,C]
    -params: labels: 真实标签，shape=[N,1]
    输出：准确率，shape=[1]
    """
    # 判断是否为二分类
    if preds.shape[1] == 1:
        # 二分类时，判断每个概率值是否大于0.5，当大于0.5时类别为1，否则类别为0
        preds = paddle.cast((preds >= 0.5), dtype='float32')
    else:
        preds = paddle.argmax(preds, axis=1, dtype='int32')
    return paddle.mean(paddle.cast(paddle.equal(preds, labels), dtype='float32'))


if __name__ == '__main__':
    preds = paddle.to_tensor([[0.], [1.], [1.], [0.]])
    labels = paddle.to_tensor([[1.], [1.], [0.], [0.]])
    print('accuracy is:', accuracy(preds, labels))

