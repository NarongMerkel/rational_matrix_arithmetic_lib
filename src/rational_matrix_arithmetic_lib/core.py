from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Union, Iterator


class MatrixDimensionError(ValueError):
    """Raised when an operation is attempted on incompatible-dimension matrices."""


class MatrixSingularError(ValueError):
    """Raised when an inverse or determinant is requested on a non-invertible matrix."""


class RationalMatrix:
    """
    A dense matrix with exact rational entries, built on Fraction.

    The whole point of using Fraction is to make intermediate cancellation
    lossless. This makes determinants and inverses of matrices that would
    be numerically catastrophic in floating point (e.g. Hilbert matrices)
    come out exact. The trade-off is that operations are much slower than
    numpy and that Fraction objects never simplify away a genuinely
    exploding numerator/denominator.
    """

    __slots__ = ("_data", "_rows", "_cols")

    def __init__(self, data: Sequence[Sequence[Union[int, Fraction, str]]]):
        rows = len(data)
        if rows == 0:
            raise MatrixDimensionError("Matrix must have at least one row.")
        cols = len(data[0])
        if cols == 0:
            raise MatrixDimensionError("Matrix must have at least one column.")

        normalized: List[List[Fraction]] = []
        for row in data:
            if len(row) != cols:
                raise MatrixDimensionError(
                    f"All rows must have the same length; expected {cols}, got {len(row)}."
                )
            normalized.append([Fraction(v) for v in row])

        self._data = normalized
        self._rows = rows
        self._cols = cols

    @classmethod
    def identity(cls, n: int) -> "RationalMatrix":
        """Construct the n-by-n identity matrix."""
        if n <= 0:
            raise MatrixDimensionError("Identity size must be a positive integer.")
        data = [[Fraction(0)] * n for _ in range(n)]
        for i in range(n):
            data[i][i] = Fraction(1)
        return cls(data)

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def shape(self) -> tuple:
        """Return (rows, cols) as a plain tuple."""
        return (self._rows, self._cols)

    def is_square(self) -> bool:
        return self._rows == self._cols

    def entry(self, i: int, j: int) -> Fraction:
        """Return the (i, j) entry as a Fraction. 0-indexed."""
        return self._data[i][j]

    def __getitem__(self, idx: tuple) -> Fraction:
        i, j = idx
        return self._data[i][j]

    def __iter__(self) -> Iterator[List[Fraction]]:
        return iter(self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RationalMatrix):
            return NotImplemented
        return self._data == other._data

    def __repr__(self) -> str:
        rows = [", ".join(str(v) for v in r) for r in self._data]
        return "RationalMatrix([[" + "], [".join(rows) + "]])"

    def to_list(self) -> List[List[Fraction]]:
        """Return a deep copy of the entries as a list of lists of Fraction."""
        return [list(r) for r in self._data]

    def __add__(self, other: "RationalMatrix") -> "RationalMatrix":
        if self.shape() != other.shape():
            raise MatrixDimensionError(
                f"Addition requires identical shapes; got {self.shape()} and {other.shape()}."
            )
        result = [
            [self._data[i][j] + other._data[i][j] for j in range(self._cols)]
            for i in range(self._rows)
        ]
        return RationalMatrix.__from_normalized(result)

    def __sub__(self, other: "RationalMatrix") -> "RationalMatrix":
        if self.shape() != other.shape():
            raise MatrixDimensionError(
                f"Subtraction requires identical shapes; got {self.shape()} and {other.shape()}."
            )
        result = [
            [self._data[i][j] - other._data[i][j] for j in range(self._cols)]
            for i in range(self._rows)
        ]
        return RationalMatrix.__from_normalized(result)

    def __mul__(self, other: Union["RationalMatrix", int, Fraction, str]) -> "RationalMatrix":
        if isinstance(other, RationalMatrix):
            return self._matmul(other)
        scalar = Fraction(other)
        result = [[v * scalar for v in row] for row in self._data]
        return RationalMatrix.__from_normalized(result)

    def __rmul__(self, scalar: Union[int, Fraction, str]) -> "RationalMatrix":
        return self.__mul__(scalar)

    def _matmul(self, other: "RationalMatrix") -> "RationalMatrix":
        if self._cols != other._rows:
            raise MatrixDimensionError(
                f"Multiplication requires self.cols == other.rows; "
                f"got {self.shape()} and {other.shape()}."
            )
        n, m, p = self._rows, self._cols, other._cols
        result = [[Fraction(0)] * p for _ in range(n)]
        for i in range(n):
            a_row = self._data[i]
            out_row = result[i]
            for k in range(m):
                aik = a_row[k]
                if aik == 0:
                    continue
                b_row = other._data[k]
                for j in range(p):
                    out_row[j] += aik * b_row[j]
        return RationalMatrix.__from_normalized(result)

    @classmethod
    def __from_normalized(cls, data: List[List[Fraction]]) -> "RationalMatrix":
        obj = cls.__new__(cls)
        obj._data = data
        obj._rows = len(data)
        obj._cols = len(data[0])
        return obj

    def determinant(self) -> Fraction:
        """
        Compute the determinant via Gaussian elimination with partial pivoting
        over the rationals. We track a sign flip for row swaps.

        Partial pivoting is not numerically motivated here (Fraction is exact),
        but it keeps the intermediate entries from growing as fast by choosing
        the row whose pivot is numerically largest at each step — a useful
        heuristic for keeping numerator/denominator sizes manageable.
        """
        if not self.is_square():
            raise MatrixDimensionError(
                f"Determinant requires a square matrix; got {self.shape()}."
            )
        n = self._rows
        # Deep copy of working matrix.
        work = [list(row) for row in self._data]
        det = Fraction(1)
        for col in range(n):
            pivot_row = -1
            best = Fraction(0)
            for r in range(col, n):
                val = work[r][col]
                if val != 0 and abs(val) > abs(best):
                    best = val
                    pivot_row = r
            if pivot_row == -1:
                return Fraction(0)
            if pivot_row != col:
                work[col], work[pivot_row] = work[pivot_row], work[col]
                det = -det
            pivot = work[col][col]
            det *= pivot
            # Eliminate below.
            for r in range(col + 1, n):
                factor = work[r][col] / pivot
                if factor == 0:
                    continue
                work_r = work[r]
                work_c = work[col]
                for c in range(col, n):
                    work_r[c] -= factor * work_c[c]
        return det

    def inverse(self) -> "RationalMatrix":
        """
        Compute the inverse by Gauss–Jordan elimination on the augmented
        matrix [A | I], with the same partial-pivoting heuristic used in
        determinant().

        Raises MatrixSingularError if no non-zero pivot can be found at any
        step. This is exact, not approximate: a matrix is invertible here if
        and only if its determinant is non-zero, and the returned inverse is
        bit-for-bit exact over the rationals.
        """
        if not self.is_square():
            raise MatrixDimensionError(
                f"Inverse requires a square matrix; got {self.shape()}."
            )
        n = self._rows
        aug = [list(self._data[i]) + [Fraction(1) if i == j else Fraction(0) for j in range(n)]
               for i in range(n)]
        for col in range(n):
            pivot_row = -1
            best = Fraction(0)
            for r in range(col, n):
                val = aug[r][col]
                if val != 0 and abs(val) > abs(best):
                    best = val
                    pivot_row = r
            if pivot_row == -1:
                raise MatrixSingularError("Matrix is singular; no inverse exists.")
            if pivot_row != col:
                aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
            pivot = aug[col][col]
            # Normalize pivot row.
            inv_pivot = Fraction(1) / pivot
            aug[col] = [v * inv_pivot for v in aug[col]]
            # Eliminate all other rows.
            for r in range(n):
                if r == col:
                    continue
                factor = aug[r][col]
                if factor == 0:
                    continue
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col])]
        result = [[aug[i][n + j] for j in range(n)] for i in range(n)]
        return RationalMatrix.__from_normalized(result)
