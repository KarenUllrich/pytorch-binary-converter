# Binary Converter

This is a tool to turn pytorch's floats into binary tensors and back.
This code converts tensors of floats or bits into the respective other.
We use the IEEE-754 guideline [1] to convert. The default for conversion are
based on 32 bit / single precision floats: 8 exponent bits and 23 mantissa bits.
Other common formats are


|num total bits    | precision   | exponent bits |  mantissa bits   |    bias |
|------------|-------------------|-------------------|-------------------|-------------------|  
|64 bits     |    double         |     11         |    52       |    1023|
|    32 bits  |       single      |         8      |       23    |        127|
|    16 bits  |       half         |        5       |      10     |        15|



### Usage

To turn a float tensor into a binary one

    from binary_converter import float2bit
    binary_tensor = float2bit(float_tensor, num_e_bits=8, num_m_bits=23, bias=127.)

To turn a binary tensor into a float one

    from binary_converter import bit2float
    float_tensor = bit2float(binary_tensor, num_e_bits=8, num_m_bits=23, bias=127.)


### Requirements

This code has been tested with
-   `python 3.6`
-   `pytorch 1.1.0`

### Maintenance

Please be warned that this repository is not going to be maintained regularly.


### References

[1] IEEE Computer Society (2008-08-29). IEEE Standard for Floating-Point
Arithmetic. IEEE Std 754-2008. IEEE. pp. 1–70. doi:10.1109/IEEESTD.2008.4610935.
ISBN 978-0-7381-5753-5. IEEE Std 754-2008
