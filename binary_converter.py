#!/usr/bin/env python3
"""Converting between floats and binaries.


This code converts tensors of floats or bits into the respective other.
We use the IEEE-754 guideline [1] to convert. The default for conversion are
based on 32 bit / single precision floats: 8 exponent bits and 23 mantissa bits.
Other common formats are


num total bits     precision    exponent bits   mantissa bits       bias
---------------------------------------------------------------------------
    64 bits         double              11             52           1023
    32 bits         single               8             23            127
    16 bits         half                 5             10             15

Available modules:
    * bit2float
    * float2bit
    * integer2bit
    * remainder2bit

[1] IEEE Computer Society (2008-08-29). IEEE Standard for Floating-Point
Arithmetic. IEEE Std 754-2008. IEEE. pp. 1–70. doi:10.1109/IEEESTD.2008.4610935.
ISBN 978-0-7381-5753-5. IEEE Std 754-2008

Author, Karen Ullrich June 2019
"""

import torch
import warnings


def bit2float(b, num_e_bits=8, num_m_bits=23, bias=127.):
  """Turn input tensor into float.

      Args:
          b : binary tensor. The last dimension of this tensor should be the
          the one the binary is at.
          num_e_bits : Number of exponent bits. Default: 8.
          num_m_bits : Number of mantissa bits. Default: 23.
          bias : Exponent bias/ zero offset. Default: 127.
      Returns:
          Tensor: Float tensor. Reduces last dimension.

  """
  expected_last_dim = num_m_bits + num_e_bits + 1
  assert b.shape[-1] == expected_last_dim, "Binary tensors last dimension " \
                                           "should be {}, not {}.".format(
    expected_last_dim, b.shape[-1])

  # check if we got the right type
  dtype = torch.float32
  if expected_last_dim > 32: dtype = torch.float64
  if expected_last_dim > 64:
    warnings.warn("pytorch can not process floats larger than 64 bits, keep"
                  " this in mind. Your result will be not exact.")

  s = torch.index_select(b, -1, torch.arange(0, 1))
  e = torch.index_select(b, -1, torch.arange(1, 1 + num_e_bits))
  m = torch.index_select(b, -1, torch.arange(1 + num_e_bits,
                                             1 + num_e_bits + num_m_bits))
  # SIGN BIT
  out = ((-1) ** s).squeeze(-1).type(dtype)
  # EXPONENT BIT
  exponents = -torch.arange(-(num_e_bits - 1.), 1.)
  exponents = exponents.repeat(b.shape[:-1] + (1,))
  e_decimal = torch.sum(e * 2 ** exponents, dim=-1) - bias
  out *= 2 ** e_decimal
  # MANTISSA
  matissa = (torch.Tensor([2.]) ** (
    -torch.arange(1., num_m_bits + 1.))).repeat(
    m.shape[:-1] + (1,))
  out *= 1. + torch.sum(m * matissa, dim=-1)
  return out


def float2bit(f, num_e_bits=8, num_m_bits=23, bias=127., dtype=torch.float32):
  """Turn input tensor into binary.

      Args:
          f : float tensor.
          num_e_bits : Number of exponent bits. Default: 8.
          num_m_bits : Number of mantissa bits. Default: 23.
          bias : Exponent bias/ zero offset. Default: 127.
          dtype : This is the actual type of the tensor that is going to be
          returned. Default: torch.float32.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.

  """
  ## SIGN BIT
  s = torch.sign(f)
  f = f * s
  # turn sign into sign-bit
  s = (s * (-1) + 1.) * 0.5
  s = s.unsqueeze(-1)

  ## EXPONENT BIT
  e_scientific = torch.floor(torch.log2(f))
  e_decimal = e_scientific + bias
  e = integer2bit(e_decimal, num_bits=num_e_bits)

  ## MANTISSA
  m1 = integer2bit(f - f % 1, num_bits=num_e_bits)
  m2 = remainder2bit(f % 1, num_bits=bias)
  m = torch.cat([m1, m2], dim=-1)
  
  dtype = f.type()
  idx = torch.arange(num_m_bits).unsqueeze(0).type(dtype) \
        + (8. - e_scientific).unsqueeze(-1)
  idx = idx.long()
  m = torch.gather(m, dim=-1, index=idx)

  return torch.cat([s, e, m], dim=-1).type(dtype)


def remainder2bit(remainder, num_bits=127):
  """Turn a tensor with remainders (floats < 1) to mantissa bits.

      Args:
          remainder : torch.Tensor, tensor with remainders
          num_bits : Number of bits to specify the precision. Default: 127.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.

  """
  dtype = remainder.type()
  exponent_bits = torch.arange(num_bits).type(dtype)
  exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
  out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
  return torch.floor(2 * out)


def integer2bit(integer, num_bits=8):
  """Turn integer tensor to binary representation.

      Args:
          integer : torch.Tensor, tensor with integers
          num_bits : Number of bits to specify the precision. Default: 8.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.

  """
  dtype = integer.type()
  exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
  exponent_bits = exponent_bits.repeat(integer.shape + (1,))
  out = integer.unsqueeze(-1) / 2 ** exponent_bits
  return (out - (out % 1)) % 2
