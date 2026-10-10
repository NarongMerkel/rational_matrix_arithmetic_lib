import unittest
from fractions import Fraction

from rational_matrix_arithmetic_lib import (
    RationalMatrix,
    MatrixDimensionError,
    MatrixSingularError,
)


class TestConstruction(unittest.TestCase):
    def test_accepts_int_fraction_str(self):
        m = RationalMatrix([[1, Fraction(1, 2), "2/3"]])
        self.assertEqual(m.entry(0, 0), Fraction(1))
        self.assertEqual(m.entry(0, 1), Fraction(1, 2))
        self.assertEqual(m.entry(0, 2), Fraction(2, 3))

    def test_empty_rows_rejected(self):
        with self.assertRaises(MatrixDimensionError):
            RationalMatrix([])

    def test_empty_columns_rejected(self):
        with self.assertRaises(MatrixDimensionError):
            RationalMatrix([[]])

    def test_ragged_rows_rejected(self):
        with self.assertRaises(MatrixDimensionError):
            RationalMatrix([[1, 2], [3]])

    def test_identity(self):
        I = RationalMatrix.identity(3)
        self.assertEqual(I.shape(), (3, 3))
        for i in range(3):
            for j in range(3):
                self.assertEqual(I.entry(i, j), Fraction(1 if i == j else 0))

    def test_identity_size_validation(self):
        with self.assertRaises(MatrixDimensionError):
            RationalMatrix.identity(0)
        with self.assertRaises(MatrixDimensionError):
            RationalMatrix.identity(-2)


class TestShapeAndAccess(unittest.TestCase):
    def setUp(self):
        self.m = RationalMatrix([[1, 2, 3], [4, 5, 6]])

    def test_shape(self):
        self.assertEqual(self.m.shape(), (2, 3))

    def test_rows_cols_props(self):
        self.assertEqual(self.m.rows, 2)
        self.assertEqual(self.m.cols, 3)

    def test_is_square_false(self):
        self.assertFalse(self.m.is_square())

    def test_is_square_true(self):
        self.assertTrue(RationalMatrix([[1, 2], [3, 4]]).is_square())

    def test_getitem(self):
        self.assertEqual(self.m[1, 2], Fraction(6))

    def test_to_list_is_deep_copy(self):
        out = self.m.to_list()
        out[0][0] = Fraction(999)
        self.assertEqual(self.m.entry(0, 0), Fraction(1))

    def test_equality(self):
        a = RationalMatrix([[1, 2], [3, 4]])
        b = RationalMatrix([[1, 2], [3, 4]])
        c = RationalMatrix([[1, 2], [3, 5]])
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class TestAddition(unittest.TestCase):
    def test_basic(self):
        a = RationalMatrix([[1, 2], [3, 4]])
        b = RationalMatrix([[5, 6], [7, 8]])
        r = a + b
        self.assertEqual(r.to_list(), [[6, 8], [10, 12]])

    def test_fraction_sum_stays_exact(self):
        a = RationalMatrix([[Fraction(1, 3)]])
        b = RationalMatrix([[Fraction(2, 3)]])
        self.assertEqual((a + b).entry(0, 0), Fraction(1))

    def test_subtraction(self):
        a = RationalMatrix([[5, 6], [7, 8]])
        b = RationalMatrix([[1, 2], [3, 4]])
        self.assertEqual((a - b).to_list(), [[4, 4], [4, 4]])

    def test_shape_mismatch_raises(self):
        a = RationalMatrix([[1, 2]])
        b = RationalMatrix([[1], [2]])
        with self.assertRaises(MatrixDimensionError):
            a + b
        with self.assertRaises(MatrixDimensionError):
            a - b


class TestMultiplication(unittest.TestCase):
    def test_basic_matmul(self):
        a = RationalMatrix([[1, 2], [3, 4]])
        b = RationalMatrix([[5, 6], [7, 8]])
        r = a * b
        self.assertEqual(r.to_list(), [[19, 22], [43, 50]])

    def test_non_square_matmul(self):
        a = RationalMatrix([[1, 2, 3], [4, 5, 6]])  # 2x3
        b = RationalMatrix([[1, 0], [0, 1], [1, 1]])  # 3x2
        r = a * b
        self.assertEqual(r.shape(), (2, 2))
        self.assertEqual(r.to_list(), [[4, 5], [10, 11]])

    def test_identity_times_matrix(self):
        m = RationalMatrix([[1, 2], [3, 4]])
        I = RationalMatrix.identity(2)
        self.assertEqual(I * m, m)
        self.assertEqual(m * I, m)

    def test_left_scalar_mul(self):
        m = RationalMatrix([[1, 2], [3, 4]])
        r = m * 3
        self.assertEqual(r.to_list(), [[3, 6], [9, 12]])

    def test_right_scalar_mul(self):
        m = RationalMatrix([[1, 2], [3, 4]])
        r = 3 * m
        self.assertEqual(r.to_list(), [[3, 6], [9, 12]])

    def test_fraction_scalar_mul(self):
        m = RationalMatrix([[1, 2], [3, 4]])
        r = m * Fraction(1, 2)
        self.assertEqual(r.to_list(), [[Fraction(1, 2), Fraction(1)], [Fraction(3, 2), Fraction(2)]])

    def test_dim_mismatch_raises(self):
        a = RationalMatrix([[1, 2]])
        b = RationalMatrix([[1, 2]])
        with self.assertRaises(MatrixDimensionError):
            a * b


class TestDeterminant(unittest.TestCase):
    def test_1x1(self):
        self.assertEqual(RationalMatrix([[7]]).determinant(), Fraction(7))

    def test_2x2(self):
        m = RationalMatrix([[1, 2], [3, 4]])
        self.assertEqual(m.determinant(), Fraction(-2))

    def test_3x3(self):
        m = RationalMatrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
        self.assertEqual(m.determinant(), Fraction(-306))

    def test_singular_zero_det(self):
        m = RationalMatrix([[1, 2, 3], [2, 4, 6], [1, 1, 1]])
        self.assertEqual(m.determinant(), Fraction(0))

    def test_hilbert_3_exact(self):
        h = RationalMatrix([
            [1, Fraction(1, 2), Fraction(1, 3)],
            [Fraction(1, 2), Fraction(1, 3), Fraction(1, 4)],
            [Fraction(1, 3), Fraction(1, 4), Fraction(1, 5)],
        ])
        self.assertEqual(h.determinant(), Fraction(1, 2160))

    def test_non_square_raises(self):
        m = RationalMatrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(MatrixDimensionError):
            m.determinant()


class TestInverse(unittest.TestCase):
    def test_2x2(self):
        m = RationalMatrix([[4, 7], [2, 6]])
        inv = m.inverse()
        expected = RationalMatrix([[Fraction(3, 5), Fraction(-7, 10)], [Fraction(-1, 5), Fraction(2, 5)]])
        self.assertEqual(inv, expected)

    def test_times_inverse_is_identity(self):
        m = RationalMatrix([[4, 7], [2, 6]])
        inv = m.inverse()
        self.assertEqual(m * inv, RationalMatrix.identity(2))
        self.assertEqual(inv * m, RationalMatrix.identity(2))

    def test_identity_inverse_is_identity(self):
        self.assertEqual(RationalMatrix.identity(3).inverse(), RationalMatrix.identity(3))

    def test_singular_raises(self):
        m = RationalMatrix([[1, 2], [2, 4]])
        with self.assertRaises(MatrixSingularError):
            m.inverse()

    def test_singular_with_row_swap_raises(self):
        m = RationalMatrix([[0, 1, 1], [0, 2, 3], [0, 5, 8]])
        with self.assertRaises(MatrixSingularError):
            m.inverse()

    def test_hilbert_inverse_exact(self):
        h = RationalMatrix([
            [1, Fraction(1, 2), Fraction(1, 3)],
            [Fraction(1, 2), Fraction(1, 3), Fraction(1, 4)],
            [Fraction(1, 3), Fraction(1, 4), Fraction(1, 5)],
        ])
        inv = h.inverse()
        self.assertEqual(h * inv, RationalMatrix.identity(3))

    def test_non_square_raises(self):
        m = RationalMatrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(MatrixDimensionError):
            m.inverse()


if __name__ == "__main__":
    unittest.main()
