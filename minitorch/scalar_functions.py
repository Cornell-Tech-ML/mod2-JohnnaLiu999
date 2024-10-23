from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Compute the backward pass for the scalar function.

        Args:
        ----
            ctx (Context): The context containing saved values for backward computation.
            d_out (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: Gradients with respect to the inputs.

        """
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Compute the forward pass for the scalar function.

        Args:
        ----
            ctx (Context): The context to save any necessary values for backward computation.
            *inps (float): Input values to the forward computation.

        Returns:
        -------
            float: The result of the forward computation.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the forward function and create a scalar object with its history.

        Args:
        ----
            *vals (ScalarLike): Input values, either as scalars or raw floats.

        Returns:
        -------
            Scalar: A scalar object with the result of the forward computation and history for backpropagation.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition.

        Args:
        ----
            ctx (Context): Unused.
            a (float): First operand.
            b (float): Second operand.

        Returns:
        -------
            float: The sum of the operands.

        """
        return float(a + b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition.

        Args:
        ----
            ctx (Context): Unused.
            d_output (float): Gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: Gradients for each input, both equal to d_output.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for logarithm.

        Args:
        ----
            ctx (Context): The context to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The logarithm of the input.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for logarithm.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): Gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.
class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication.

        Args:
        ----
            ctx (Context): The context to save values for backward computation.
            a (float): First operand.
            b (float): Second operand.

        Returns:
        -------
            float: The product of the operands.

        """
        ctx.save_for_backward(a, b)
        return float(a * b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): Gradient of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to both inputs.

        """
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inversion.

        Args:
        ----
            ctx (Context): The context to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The inverse of the input.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inversion.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): Gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation.

        Args:
        ----
            ctx (Context): Unused.
            a (float): The input value.

        Returns:
        -------
            float: The negation of the input.

        """
        return -1.0 * a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation.

        Args:
        ----
            ctx (Context): Unused.
            d_output (float): Gradient of the output.

        Returns:
        -------
            float: The negation of the gradient.

        """
        return -1.0 * d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function f(x) = 1 / (1 + exp(-x))."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid.

        Args:
        ----
            ctx (Context): The context to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of the sigmoid function.

        """
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): Gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        sigmoid_a = operators.sigmoid(a)
        return sigmoid_a * (1.0 - sigmoid_a) * d_output


class ReLU(ScalarFunction):
    """ReLU function f(x) = max(0, x)."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU.

        Args:
        ----
            ctx (Context): The context to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of the ReLU function.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): Gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function f(x) = exp(x)."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponentiation.

        Args:
        ----
            ctx (Context): The context to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of the exponential function.

        """
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exponentiation.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): Gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.exp(a) * d_output


class LT(ScalarFunction):
    """Less-than function $f(a, b) =$ 1.0 if a is less than b else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less-than comparison.

        Args:
        ----
            ctx (Context): Unused.
            a (float): First operand.
            b (float): Second operand.

        Returns:
        -------
            float: 1.0 if a is less than b, else 0.0.

        """
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less-than comparison.

        Args:
        ----
            ctx (Context): Unused.
            d_output (float): Gradient of the output.

        Returns:
        -------
            Tuple[float, float]: Zero gradients, as the less-than operation is not differentiable.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(a,b) =$ 1.0 if a is equal to b else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equality comparison.

        Args:
        ----
            ctx (Context): Unused.
            a (float): First operand.
            b (float): Second operand.

        Returns:
        -------
            float: 1.0 if a is equal to b, else 0.0.

        """
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradient of the equal function.

        Args:
        ----
            ctx (Context): The context containing saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to the inputs a and b (both 0.0).

        """
        return 0.0, 0.0
