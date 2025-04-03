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
}

impl Matrix<2, 2> {
    pub fn determinant(&self) -> f32 {
        self.elements[0][0] * self.elements[1][1] - self.elements[0][1] * self.elements[1][0]
    }
}

impl Matrix<3, 3> {
    pub fn determinant(&self) -> f32 {
        let [e11, e12, e13] = self.elements[0];
        let [e21, e22, e23] = self.elements[1];
        let [e31, e32, e33] = self.elements[2];

        e11 * (e22 * e33 - e23 * e32)
            + e12 * (e23 * e31 - e21 * e33)
            + e13 * (e21 * e32 - e22 * e31)
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
}
