#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Sub;
use std::ops::SubAssign;

#[derive(Debug, PartialEq)]
pub struct Matrix<const M: usize, const N: usize> {
    pub elements: [[f32; N]; M],
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    pub fn new(elements: [[f32; N]; M]) -> Self {
        Matrix { elements }
    }

    pub fn transpose(&self) -> Matrix<N, M> {
        let mut elements = [[0.0; M]; N];
        for m in 0..M {
            for n in 0..N {
                elements[n][m] = self.elements[m][n];
            }
        }
        Matrix { elements }
    }

    pub fn row_column_num(&self) -> (usize, usize) {
        (M, N)
    }
}

impl<const M: usize> Matrix<M, M> {
    pub fn submatrix(
        &self,
        exclude_row_index: usize,
        exclude_column_index: usize,
    ) -> Matrix<{ M - 1 }, { M - 1 }> {
        let (row_num, column_num) = self.row_column_num();

        let remained_elements = self
            .elements
            .iter()
            .flatten()
            .enumerate()
            .filter(|(i, _e)| {
                i / column_num != exclude_row_index && i % column_num != exclude_column_index
            })
            .map(|(_i, e)| e)
            .collect::<Vec<_>>();
        let chunked_elements = remained_elements.chunks(column_num - 1).collect::<Vec<_>>();

        let mut elements = [[0.0; M - 1]; M - 1];
        for row_index in 0..row_num - 1 {
            for column_index in 0..column_num - 1 {
                elements[row_index][column_index] = *chunked_elements[row_index][column_index];
            }
        }

        Matrix { elements }
    }

    /// determinant of submatrix
    pub fn minor(&self, exclude_row_index: usize, exclude_column_index: usize) -> f32
    where
        [(); M - 1]:,
    {
        let submatrix = self.submatrix(exclude_row_index, exclude_column_index);
        submatrix.determinant()
    }

    pub fn cofactor(&self, row_index: usize, column_index: usize) -> f32
    where
        [(); M - 1]:,
    {
        let is_plus_sign = (row_index + column_index) % 2 == 0;
        let sign = if is_plus_sign { 1.0 } else { -1.0 };

        let minor = self.minor(row_index, column_index);

        let result = sign * minor;
        result
    }

    // FIXME
    pub fn determinant(&self) -> f32 {
        match M {
            2 => self.determinant_2x2(),
            3 => self.determinant_3x3(),
            4 => self.determinant_4x4(),
            _ => unimplemented!(),
        }
    }

    fn determinant_2x2(&self) -> f32 {
        self.elements[0][0] * self.elements[1][1] - self.elements[0][1] * self.elements[1][0]
    }

    // FIXME
    fn determinant_3x3(&self) -> f32 {
        let e11 = self.elements[0][0];
        let e12 = self.elements[0][1];
        let e13 = self.elements[0][2];

        let e21 = self.elements[1][0];
        let e22 = self.elements[1][1];
        let e23 = self.elements[1][2];

        let e31 = self.elements[2][0];
        let e32 = self.elements[2][1];
        let e33 = self.elements[2][2];

        e11 * (e22 * e33 - e23 * e32)
            + e12 * (e23 * e31 - e21 * e33)
            + e13 * (e21 * e32 - e22 * e31)
    }

    // FIXME
    fn determinant_4x4(&self) -> f32 {
        let e11 = self.elements[0][0];
        let e12 = self.elements[0][1];
        let e13 = self.elements[0][2];
        let e14 = self.elements[0][3];

        let e21 = self.elements[1][0];
        let e22 = self.elements[1][1];
        let e23 = self.elements[1][2];
        let e24 = self.elements[1][3];

        let e31 = self.elements[2][0];
        let e32 = self.elements[2][1];
        let e33 = self.elements[2][2];
        let e34 = self.elements[2][3];

        let e41 = self.elements[3][0];
        let e42 = self.elements[3][1];
        let e43 = self.elements[3][2];
        let e44 = self.elements[3][3];

        e11 * (e22 * (e33 * e44 - e34 * e43)
            + e23 * (e34 * e42 - e32 * e44)
            + e24 * (e32 * e43 - e33 * e42))
            - e12
                * (e21 * (e33 * e44 - e34 * e43)
                    + e23 * (e34 * e41 - e31 * e44)
                    + e24 * (e31 * e43 - e33 * e41))
            + e13
                * (e21 * (e32 * e44 - e34 * e42)
                    + e22 * (e34 * e41 - e31 * e44)
                    + e24 * (e31 * e42 - e32 * e41))
            - e14
                * (e21 * (e32 * e43 - e33 * e42)
                    + e22 * (e33 * e41 - e31 * e43)
                    + e23 * (e31 * e42 - e32 * e41))
    }

    pub fn adjoint(&self) -> Matrix<M, M>
    where
        [(); M - 1]:,
    {
        let mut elements = [[0.0; M]; M];

        for row in 0..M {
            for column in 0..M {
                let cofactor = self.cofactor(row, column);
                elements[row][column] = cofactor;
            }
        }

        let matrix = Matrix { elements };
        matrix.transpose()
    }

    pub fn invert(&self) -> Matrix<M, M>
    where
        [(); M - 1]:,
    {
        let adjoint = self.adjoint();
        let determinant = self.determinant();
        adjoint * (1.0 / determinant)
    }
}

impl<const M: usize, const N: usize> Add for Matrix<M, N> {
    type Output = Matrix<M, N>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut elements = [[0.0; N]; M];
        for m in 0..M {
            for n in 0..N {
                elements[m][n] = self.elements[m][n] + rhs.elements[m][n];
            }
        }
        return Matrix { elements };
    }
}

impl<const M: usize, const N: usize> AddAssign for Matrix<M, N> {
    fn add_assign(&mut self, rhs: Self) {
        for m in 0..M {
            for n in 0..N {
                self.elements[m][n] += rhs.elements[m][n];
            }
        }
    }
}

impl<const M: usize, const N: usize> Sub for Matrix<M, N> {
    type Output = Matrix<M, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut elements = [[0.0; N]; M];
        for m in 0..M {
            for n in 0..N {
                elements[m][n] = self.elements[m][n] - rhs.elements[m][n];
            }
        }
        return Matrix { elements };
    }
}

impl<const M: usize, const N: usize> SubAssign for Matrix<M, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for m in 0..M {
            for n in 0..N {
                self.elements[m][n] -= rhs.elements[m][n];
            }
        }
    }
}

impl<const M: usize, const N: usize, const P: usize> Mul<Matrix<N, P>> for Matrix<M, N> {
    type Output = Matrix<M, P>;

    fn mul(self, rhs: Matrix<N, P>) -> Self::Output {
        let mut elements = [[0.0; P]; M];
        for row in 0..M {
            for column in 0..P {
                for i in 0..N {
                    elements[row][column] += self.elements[row][i] * rhs.elements[i][column];
                }
            }
        }
        return Matrix { elements };
    }
}

impl<const M: usize, const N: usize> Mul<f32> for Matrix<M, N> {
    type Output = Matrix<M, N>;

    fn mul(self, scalar: f32) -> Self::Output {
        let mut elements = self.elements.clone();
        for m in 0..M {
            for n in 0..N {
                elements[m][n] *= scalar;
            }
        }
        return Matrix { elements };
    }
}

impl<const M: usize, const N: usize> MulAssign<Matrix<N, N>> for Matrix<M, N> {
    fn mul_assign(&mut self, rhs: Matrix<N, N>) {
        for row in 0..M {
            let original_row = self.elements[row].clone();
            for column in 0..N {
                let mut result = 0.0;
                for i in 0..N {
                    result += original_row[i] * rhs.elements[i][column];
                }
                self.elements[row][column] = result;
            }
        }
    }
}

impl<const M: usize, const N: usize> MulAssign<f32> for Matrix<M, N> {
    fn mul_assign(&mut self, scalar: f32) {
        for m in 0..M {
            for n in 0..N {
                self.elements[m][n] *= scalar;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Matrix;

    fn near_zero(value: f32, epsilon: f32) -> bool {
        value.abs() <= epsilon
    }

    #[macro_export]
    macro_rules! assert_near_eq {
        ($left:expr, $right:expr, $epsilon:expr $(,)?) => {
            match (&$left, &$right, &$epsilon) {
                (left_val, right_val, epsilon_val) => {
                    assert!(
                        near_zero(*left_val - *right_val, *epsilon_val),
                        "`left == right` failed... left = {}, right = {}",
                        *left_val,
                        *right_val
                    );
                }
            }
        };
    }

    #[test]
    fn add_test() {
        let expected: Matrix<2, 2> = Matrix::new([[3.0, 10.0], [10.0, -8.0]]);

        let a: Matrix<2, 2> = Matrix::new([[3.0, 7.0], [6.0, -4.0]]);
        let b: Matrix<2, 2> = Matrix::new([[0.0, 3.0], [4.0, -4.0]]);
        let actual: Matrix<2, 2> = a + b;

        // compilation error
        // let c: Matrix<2, 3> = Matrix::new([[0.0; 3], [0.0; 3]]);
        // a + c;

        assert_eq!(actual, expected);
    }

    #[test]
    fn add_assign_test() {
        let expected: Matrix<2, 2> = Matrix::new([[3.0, 10.0], [10.0, -8.0]]);

        let mut a: Matrix<2, 2> = Matrix::new([[3.0, 7.0], [6.0, -4.0]]);
        let b: Matrix<2, 2> = Matrix::new([[0.0, 3.0], [4.0, -4.0]]);
        a += b;

        // compilation error
        // let c: Matrix<2, 3> = Matrix::new([[0.0; 3], [0.0; 3]]);
        // a += c;

        assert_eq!(a, expected);
    }

    #[test]
    fn sub_test() {
        let expected: Matrix<2, 2> = Matrix::new([[4.0, -1.0], [-5.0, 1.0]]);

        let a: Matrix<2, 2> = Matrix::new([[2.0, 0.0], [-1.0, 4.0]]);
        let b: Matrix<2, 2> = Matrix::new([[-2.0, 1.0], [4.0, 3.0]]);
        let actual: Matrix<2, 2> = a - b;

        // compilation error
        // let c: Matrix<1, 2> = Matrix::new([[1.0, 2.0]]);
        // a - c;

        assert_eq!(actual, expected);
    }

    #[test]
    fn sub_assign_test() {
        let expected: Matrix<2, 2> = Matrix::new([[4.0, -1.0], [-5.0, 1.0]]);

        let mut a: Matrix<2, 2> = Matrix::new([[2.0, 0.0], [-1.0, 4.0]]);
        let b: Matrix<2, 2> = Matrix::new([[-2.0, 1.0], [4.0, 3.0]]);
        a -= b;

        // compilation error
        // let c: Matrix<1, 2> = Matrix::new([[1.0, 2.0]]);
        // a -= c;

        assert_eq!(a, expected);
    }

    #[test]
    fn multiply_scalar_test() {
        let expected: Matrix<1, 3> = Matrix::new([[3.0, 6.0, 9.0]]);

        let a: Matrix<1, 3> = Matrix::new([[1.0, 2.0, 3.0]]);
        let actual: Matrix<1, 3> = a * 3.0;

        assert_eq!(actual, expected);
    }

    #[test]
    fn multiply_other_test() {
        let expected: Matrix<3, 3> =
            Matrix::new([[5.0, 11.0, 7.0], [5.0, 13.0, 16.0], [7.0, 17.0, 17.0]]);

        let a: Matrix<3, 2> = Matrix::new([[1.0, 2.0], [3.0, 1.0], [3.0, 2.0]]);
        let b: Matrix<2, 3> = Matrix::new([[1.0, 3.0, 5.0], [2.0, 4.0, 1.0]]);
        let actual: Matrix<3, 3> = a * b;

        // compilation error
        // let c: Matrix<1, 2> = Matrix::new([[3.0, 7.0]]);
        // a * c;

        assert_eq!(actual, expected);
    }

    #[test]
    fn multiply_assign_scalar_test() {
        let expected: Matrix<1, 3> = Matrix::new([[3.0, 6.0, 9.0]]);

        let mut a: Matrix<1, 3> = Matrix::new([[1.0, 2.0, 3.0]]);
        a *= 3.0;

        assert_eq!(a, expected);
    }

    #[test]
    fn multiply_assign_other_test() {
        let expected: Matrix<3, 2> = Matrix::new([[5.0, 11.0], [5.0, 13.0], [7.0, 17.0]]);

        let mut a: Matrix<3, 2> = Matrix::new([[1.0, 2.0], [3.0, 1.0], [3.0, 2.0]]);
        let b: Matrix<2, 2> = Matrix::new([[1.0, 3.0], [2.0, 4.0]]);
        a *= b;

        // compilation error
        // let c: Matrix<2, 3> = Matrix::new([[1.0, 3.0, 5.0], [2.0, 4.0, 1.0]]);
        // a *= c;

        assert_eq!(a, expected);
    }

    #[test]
    fn transpose_test() {
        let expected: Matrix<3, 2> = Matrix::new([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);

        let a: Matrix<2, 3> = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let actual: Matrix<3, 2> = a.transpose();

        assert_eq!(actual, expected);
    }

    #[test]
    fn determinant_2x2_test() {
        let expected = 5.0;

        let a = Matrix::new([[2.0, 1.0], [-1.0, 2.0]]);
        let actual = a.determinant();

        assert_eq!(actual, expected);
    }

    #[test]
    fn determinant_3x3_test() {
        let expected = 22.0;

        let a = Matrix::new([[3.0, -2.0, 0.0], [1.0, 4.0, -3.0], [-1.0, 0.0, 2.0]]);
        let actual = a.determinant();

        assert_eq!(actual, expected);
    }

    #[test]
    fn determinant_4x4_test() {
        let expected = -16.0;

        let a = Matrix::new([
            [1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0],
        ]);
        let actual = a.determinant();

        assert_eq!(actual, expected);
    }

    #[test]
    fn submatrix_2x2_test() {
        let expected = Matrix::new([[-1.0]]);

        let a = Matrix::new([[2.0, 1.0], [-1.0, 2.0]]);
        let actual = a.submatrix(0, 1);

        assert_eq!(actual, expected);
    }

    #[test]
    fn submatrix_3x3_test() {
        let expected = Matrix::new([[0.0, -2.0], [1.0, -1.0]]);

        let a = Matrix::new([[-4.0, -3.0, 3.0], [0.0, 2.0, -2.0], [1.0, 4.0, -1.0]]);
        let actual = a.submatrix(0, 1);

        assert_eq!(actual, expected);
    }

    #[test]
    fn cofactor_3x3_test() {
        let expected1 = 6.0;
        let expected2 = 13.0;
        let expected3 = -8.0;

        let a = Matrix::new([[-4.0, -3.0, 3.0], [0.0, 2.0, -2.0], [1.0, 4.0, -1.0]]);
        let actual1 = a.cofactor(0, 0);
        let actual2 = a.cofactor(1, 2);
        let actual3 = a.cofactor(2, 2);

        assert_eq!(actual1, expected1);
        assert_eq!(actual2, expected2);
        assert_eq!(actual3, expected3);
    }

    #[test]
    fn adjoint_test() {
        let expected = Matrix::new([[6.0, 9.0, 0.0], [-2.0, 1.0, -8.0], [-2.0, 13.0, -8.0]]);

        let a = Matrix::new([[-4.0, -3.0, 3.0], [0.0, 2.0, -2.0], [1.0, 4.0, -1.0]]);
        let actual = a.adjoint();

        assert_eq!(actual, expected);
    }

    #[test]
    fn invert_test() {
        let expected = Matrix::new([
            [-1.0 / 4.0, -3.0 / 8.0, 0.0],
            [1.0 / 12.0, -1.0 / 24.0, 1.0 / 3.0],
            [1.0 / 12.0, -13.0 / 24.0, 1.0 / 3.0],
        ]);

        let a = Matrix::new([[-4.0, -3.0, 3.0], [0.0, 2.0, -2.0], [1.0, 4.0, -1.0]]);
        let actual = a.invert();

        assert_eq!(actual, expected);
    }

    #[test]
    fn learning_invert_test() {
        // Identity matrix
        let expected = Matrix::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

        // Multiplying original matrix and invert Matrix makes identity matrix
        let a = Matrix::new([[-4.0, -3.0, 3.0], [0.0, 2.0, -2.0], [1.0, 4.0, -1.0]]);
        let invert = a.invert();
        let actual = a * invert;

        for row in 0..3 {
            for column in 0..3 {
                let expected = expected.elements[row][column];
                let actual = actual.elements[row][column];
                assert_near_eq!(actual, expected, 0.000001);
            }
        }
    }
}
