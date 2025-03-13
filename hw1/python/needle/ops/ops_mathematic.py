"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return array_api.power(a, b)
        
    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad * b * power(a, b - 1), out_grad * power(a, b) * log(a)

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * self.scalar * power_scalar(a, self.scalar - 1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a:NDArray, b:NDArray):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad / b, -out_grad * a / (b * b)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar
    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return array_api.swapaxes(a, -1, -2)
        else:
            return array_api.swapaxes(a, *self.axes)

    def gradient(self, out_grad, node):
        return transpose(out_grad, axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    # 改变维度，不改变数据(重新组织数据)
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)

def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        # 先归约多余的维度，因为broadcast就是把维度扩展到shape的维度再把1对齐
        ret = summation(out_grad, tuple(range(len(out_grad.shape) - len(input_shape))))
        for i, dim in enumerate(input_shape):
            if dim == 1:
              # 原维度为1则继续归约
              ret = summation(ret, axes=(i,))
        # 因为把1对齐了，这个维度被归约了，因此要还原
        return reshape(ret, input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        # 通过广播机制，把out_grad的维度还原到input_node的维度
        input_node = node.inputs[0]
        axes = self.axes
        broadcast_axes = list(input_node.shape)
        if axes is None:
            # 默认在第一维度上进行归约
            axes = list(range(len(input_node.shape)))
        # 归约的维度设定为1
        for i in axes:
            broadcast_axes[i] = 1
        # 先把1补了，因为1是广播的维度，是离散的，直接广播不一定是期望的行为
        return broadcast_to(reshape(out_grad, broadcast_axes), input_node.shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a@b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        # 要把广播的维度还原，进行归约操作，可以推导一些简单的情形，把多余维度归约就是grad的结果
        a_p_adj = out_grad @ transpose(b)
        b_p_adj = transpose(a) @ out_grad
        # 归约操作,把多余的维度归约，和a，b相比
        a_p_adj = summation(a_p_adj, axes=tuple(range(len(a_p_adj.shape) - len(a.shape))))
        b_p_adj = summation(b_p_adj, axes=tuple(range(len(b_p_adj.shape) - len(b.shape))))
        return a_p_adj, b_p_adj


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return negate(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])

def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        # a:Ndarray
        output = array_api.copy(a)
        output[a < 0] = 0
        return output
    
    def gradient(self, out_grad, node):
        # 对out_grad逐个做max(0,x)即可
        # 先获取Ndarray
        mask = node.realize_cached_data().copy()
        mask[mask > 0] = 1
        # 不需要把小于0变成0，因为node已经经过了relu
        return out_grad * Tensor(mask)


def relu(a):
    return ReLU()(a)

