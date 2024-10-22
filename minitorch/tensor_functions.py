"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Performs the backward pass for the function.

        Args:
        ----
            ctx (Context): The context containing saved tensors for backpropagation.
            grad_out (Tensor): The gradient of the output with respect to the loss.

        Returns:
        -------
            Tuple[Tensor, ...]: The gradients of the inputs with respect to the loss.

        """
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        """Performs the forward pass for the function."""
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for negation operation.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            t1 (Tensor): Input tensor to be negated.

        Returns:
        -------
            Tensor: The negated tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for negation operation.

        Args:
        ----
            ctx (Context): Context object containing saved intermediate values.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tensor: Gradient of the loss with respect to the input.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the inverse operation.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            t1 (Tensor): Input tensor to be inverted.

        Returns:
        -------
            Tensor: The inverted tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the inverse operation.

        Args:
        ----
            ctx (Context): Context object containing saved intermediate values.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tensor: Gradient of the loss with respect to the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for the addition operation.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            t1 (Tensor): First input tensor.
            t2 (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: The result of adding the two tensors.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the addition operation.

        Args:
        ----
            ctx (Context): Context object containing saved intermediate values.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the loss with respect to the inputs.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass to check if all elements are true.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            a (Tensor): Input tensor to check.
            dim (Tensor): Dimension to reduce along.

        Returns:
        -------
            Tensor: 1 if all elements are true, otherwise 0.

        """
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for multiplication.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            t1 (Tensor): First input tensor.
            t2 (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: The result of multiplying the two tensors.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for multiplication.

        Args:
        ----
            ctx (Context): Context object containing saved intermediate values.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the loss with respect to the inputs.

        """
        (t1, t2) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, t2), grad_output.f.mul_zip(
            grad_output, t1
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Forward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            a (Tensor): Input tensor.

        Returns:
        -------
            Tensor: The result of applying the sigmoid function.

        """
        sig = a.f.sigmoid_map(a)
        ctx.save_for_backward(sig)
        return sig

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): Context object containing saved intermediate values.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tensor: Gradient of the loss with respect to the input.

        """
        (sig,) = ctx.saved_values

        # Calculate 1 - sig
        one_minus_sig = grad_output.f.add_zip(tensor(1), sig.f.neg_map(sig))

        # Calculate sig * (1 - sig)
        sigmoid_grad = grad_output.f.mul_zip(sig, one_minus_sig)

        # Multiply sigmoid_grad with grad_output
        return grad_output.f.mul_zip(sigmoid_grad, grad_output)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the ReLU function.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            t1 (Tensor): Input tensor.

        Returns:
        -------
            Tensor: The result of applying the ReLU function.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the ReLU function.

        Args:
        ----
            ctx (Context): Context object containing saved intermediate values.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tensor: Gradient of the loss with respect to the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the log function.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            t1 (Tensor): Input tensor.

        Returns:
        -------
            Tensor: The result of applying the log function.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the log function.

        Args:
        ----
            ctx (Context): Context object containing saved intermediate values.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tensor: Gradient of the loss with respect to the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the exponential function.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            t1 (Tensor): Input tensor.

        Returns:
        -------
            Tensor: The result of applying the exponential function.

        """
        ctx.save_for_backward(t1)
        return t1.f.exp_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the exponential function.

        Args:
        ----
            ctx (Context): Context object containing saved intermediate values.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tensor: Gradient of the loss with respect to the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, t1.f.exp_map(t1))


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor | None = None) -> Tensor:
        """Forward pass for the sum operation.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            t1 (Tensor): Input tensor to sum.
            dim (Tensor, optional): The dimension along which to sum. Defaults to None.

        Returns:
        -------
            Tensor: The sum of the input tensor along the specified dimension.

        """
        return t1.f.add_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the sum operation.

        Args:
        ----
            ctx (Context): Context object containing saved intermediate values.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Union[Tensor, Tuple[Tensor, Tensor]]: Gradients of the loss with respect
            to the input and the dimension, if applicable.

        """
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Forward pass for the less than operation.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            a (Tensor): First input tensor.
            b (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: A tensor with 1 where the values of `a` are less than `b`, otherwise 0.

        """
        ctx.save_for_backward(a, b)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the less than operation.

        Args:
        ----
            ctx (Context): Context object containing saved intermediate values.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the loss with respect to the inputs.

        """
        a, b = ctx.saved_values

        zeros_a = a.zeros(a.shape)
        zeros_b = b.zeros(b.shape)
        return zeros_a, zeros_b


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for the equal operation.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            t1 (Tensor): First input tensor.
            t2 (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: A tensor with 1 where the values of `t1` equal `t2`, otherwise 0.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the equal operation.

        Args:
        ----
            ctx (Context): Context object containing saved intermediate values.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the loss with respect to the inputs.

        """
        (t1, t2) = ctx.saved_values
        return zeros(t1.shape), zeros(t2.shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for the is_close operation, which checks if two tensors are close.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            t1 (Tensor): First input tensor.
            t2 (Tensor): Second input tensor.

        Returns:
        -------
            Tensor: A tensor indicating where the two tensors are close.

        """
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Forward pass for permuting the dimensions of a tensor.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            t1 (Tensor): Input tensor.
            order (Tensor): A tensor specifying the permutation order.

        Returns:
        -------
            Tensor: The permuted tensor.

        """
        # Save dims for backward pass
        ctx.save_for_backward(order)
        # Convert dims to a list
        order_list = [int(d) for d in order.to_numpy()]
        # Use the permute function of tensor
        return t1._new(t1._tensor.permute(*order_list))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the permute operation, which reverses the permutation.

        Args:
        ----
            ctx (Context): Context object containing saved intermediate values.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the loss with respect to the input and
            the permutation order.

        """
        (order,) = ctx.saved_values
        order_list = [int(d) for d in order.to_numpy()]
        # Inverse the permutation for the backward pass
        inverse_dims = [0] * len(order_list)
        for i, d in enumerate(order_list):
            inverse_dims[d] = i
        return grad_output._new(grad_output._tensor.permute(*inverse_dims)), zeros(
            order.shape
        )


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Forward pass for the view operation, reshaping the tensor.

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            a (Tensor): Input tensor.
            shape (Tensor): Target shape for the view operation.

        Returns:
        -------
            Tensor: The reshaped tensor.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the view operation, restoring the original shape.

        Args:
        ----
            ctx (Context): Context object containing saved intermediate values.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, float]: The reshaped gradient and a constant 0.0.

        """
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Forward pass for copying a tensor (makes it contiguous).

        Args:
        ----
            ctx (Context): Context object to save intermediate values.
            a (Tensor): Input tensor.

        Returns:
        -------
            Tensor: The copied tensor.

        """
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the copy operation.

        Args:
        ----
            ctx (Context): Context object containing saved intermediate values.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tensor: The gradient of the input, unchanged.

        """
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the central difference approximation of the gradient.

    Args:
    ----
        f (Any): The function whose gradient is being checked.
        vals (Tensor): The input tensors.
        arg (int, optional): The argument to compute the gradient with respect to. Defaults to 0.
        epsilon (float, optional): The small value used for finite differences. Defaults to 1e-6.
        ind (UserIndex): The index for which to compute the central difference.

    Returns:
    -------
        float: The approximated gradient at the given index.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
