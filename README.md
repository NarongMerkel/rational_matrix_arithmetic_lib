# rational_matrix_arithmetic_lib

Exact rational matrix arithmetic: addition, subtraction, multiplication, determinant, and inverse over `fractions.Fraction` entries, so results never accumulate floating-point error.

## Usage

```python
from rational_matrix_arithmetic_lib import RationalMatrix

a = RationalMatrix([[1, 2], [3, 4]])
b = RationalMatrix([[5, 6], [7, 8]])

print((a + b).to_list())          # [[6, 8], [10, 12]]
print((a * b).to_list())          # [[19, 22], [43, 50]]
print(a.determinant())            # -2
print(a.inverse().to_list())      # [[-2, 1], [3/2, -1/2]]
```

## Exported names

- `RationalMatrix` — the matrix class.
- `MatrixDimensionError` — raised on shape mismatches.
- `MatrixSingularError` — raised when an inverse is requested on a non-invertible matrix.

## Why this exists

Floating-point matrix code quietly loses precision on matrices that are ill-conditioned but mathematically clean — the canonical example being Hilbert matrices, whose entries are simple unit fractions but whose determinants are astronomically small. This library uses `fractions.Fraction` throughout so every intermediate stays exact. The trade-off is speed: `Fraction` arithmetic is many orders of magnitude slower than `float`, and numerators/denominators can grow large during elimination. This is a tool for correctness-sensitive, small-to-medium matrices, not for bulk numerics.

## Edge you will hit

`Fraction` accepts `str` like `"2/3"` as well as `int` and `Fraction` at construction time, which is convenient but means a typo like `"1 /2"` (with a stray space) will raise `ValueError` from the standard library, not from this library. Construct from `int` or `Fraction` if you want stricter input control. Singular matrices are detected during elimination, not pre-checked, so `inverse()` on a singular matrix raises `MatrixSingularError` rather than returning a sentinel.
