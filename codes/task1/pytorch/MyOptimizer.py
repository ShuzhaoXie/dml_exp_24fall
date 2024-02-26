class BaseOptimizer():
    def __init__(self, params, lr=0.001):
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


class SGDOptimizer(BaseOptimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params, lr)
        self.state = dict()
        for p in self.params:
            self.state[p] = dict()
            self.state[p]['t'] = 0 
        
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            self.state[p]['t'] += 1
            self.state[p]['m'] = p.grad
            p.data.add_(self.state[p]['m'], alpha=-self.lr)
        
            
class SGDMOptimizer(BaseOptimizer):
    # write your code here
    pass
class AdamOptimizer(BaseOptimizer):
    # write your code here
    pass