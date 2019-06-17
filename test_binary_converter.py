#!/usr/bin/env python3
"""Tests for  binary_conversion_tools.py


This code tests the conversion of tensors of floats or bits into the respective
other. We use the IEEE-754 guideline [1] to convert. The default for conversion
are based on 32 bit / single precision floats: 8 exponent bits and 23 mantissa
bits.
Other common formats are


num total bits     precision    exponent bits   mantissa bits       bias
---------------------------------------------------------------------------
    64 bits         double              11             52           1023
    32 bits         single               8             23            127
    16 bits         half                 5             10             15


The correct answers for the test have been generated with help of this website:
https://www.h-schmidt.net/FloatConverter/IEEE754.html

[1] IEEE Computer Society (2008-08-29). IEEE Standard for Floating-Point
Arithmetic. IEEE Std 754-2008. IEEE. pp. 1–70. doi:10.1109/IEEESTD.2008.4610935.
ISBN 978-0-7381-5753-5. IEEE Std 754-2008

Author, Karen Ullrich June 2019
"""
import numpy as np
import torch
from torch import nn


from binary_converter import float2bit, bit2float

# IEEE-754 SINGLE
print("Testing the float to bit conversion on IEEE-754 single precision "
      "floating point format.")

f = torch.Tensor([[123.123, 0.0003445]])

f = nn.Parameter(f)
target = torch.Tensor([[[0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0,
                         0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
                        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1,
                         0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                         0]]]).type(torch.float32)

pred = float2bit(f, num_e_bits=8, num_m_bits=23, bias=127.)

try:
  torch.sum(pred).backward()
except:
  "Function, float2bit, is not differntiable."

assert (f.shape == pred.shape[:-1]), "float2bit does not produce correct shape"

assert torch.all(pred == target), "float2bit does not produce the correct " \
                                  "result, correct solution: {}, computed " \
                                  "solution {}".format(target, pred)
print("Test successful.")

print("Testing the bit to float conversion on IEEE-754 single precision "
      "floating point format.")

b_1 = '01000010111101101110100101111001'
b_1 = [np.int(f) for f in list(b_1)]
b_2 = '00111010001010111000011010111101'
b_2 = [np.int(f) for f in list(b_2)]
b = torch.Tensor([b_1, b_2])

pred = bit2float(b)
target = torch.Tensor([123.456, 0.000654321])

assert (pred.shape == b.shape[:-1]), "bit2float does not produce correct shape"

assert torch.all(pred == target), \
  "bit2float does not produce the correct result, correct solution: {}, " \
  "computed solution {}".format(target, pred)

print("Test successful.")

