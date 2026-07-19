import torch

# y = wx+b
# ctx指上下文保存的参数值
# backward中的grad_out参数指上一次的梯度，通常用于链式法则计算
class line(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, b):
        ctx.save_for_backward(w, x, b)
        return w * x + b

    @staticmethod
    def backward(ctx, grad_out):
        w, x, b = ctx.saved_tensors
        grad_w = grad_out * x
        grad_x = grad_out * w
        grad_b = grad_out
        return grad_w, grad_x, grad_b

w = torch.rand(2, 2, requires_grad=True)
x = torch.rand(2, 2, requires_grad=True)
b = torch.rand(2, 2, requires_grad=True)

out = line.apply(w, x, b)
out.backward(torch.ones(2, 2))

print(w, x, b)
print(w.grad, x.grad, b.grad)