from mindspore import nn, ops
# from mindspore import Parameter, Tensor
# import mindspore as ms

class SGDOptimizer(nn.Optimizer):
    def __init__(self, params, lr=0.001):
        super(SGDOptimizer, self).__init__(lr, params)
    
    def construct(self, gradients):
        raise NotImplementedError
    
    
class SGDMOptimizer(nn.Optimizer):
    pass

class AdamOptimizer(nn.Optimizer):
    pass