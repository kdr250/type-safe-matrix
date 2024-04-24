use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Mul;

#[derive(Debug, PartialEq)]
pub struct Matrix2D<const M: usize, const N: usize> {
    pub elements: [[f32; N]; M],
}

impl<const M: usize, const N: usize> Matrix2D<M, N> {
    pub fn new(elements: [[f32; N]; M]) -> Self {
        Matrix2D { elements }
    }
}

impl<const M: usize, const N: usize> Add for Matrix2D<M, N> {
    type Output = Matrix2D<M, N>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut elements = [[0.0; N]; M];
        for m in 0..M {
            for n in 0..N {
                elements[m][n] = self.elements[m][n] + rhs.elements[m][n];
            }
        }
        return Matrix2D { elements };
    }
}

impl<const M: usize, const N: usize> AddAssign for Matrix2D<M, N> {
    fn add_assign(&mut self, rhs: Self) {
        for m in 0..M {
            for n in 0..N {
                self.elements[m][n] += rhs.elements[m][n];
            }
        }
    }
}

impl<const M: usize, const N: usize, const L: usize> Mul<Matrix2D<N, L>> for Matrix2D<M, N> {
    type Output = Matrix2D<M, L>;

    fn mul(self, rhs: Matrix2D<N, L>) -> Self::Output {
        let mut elements = [[0.0; L]; M];
        for row in 0..M {
            for column in 0..L {
                for i in 0..N {
                    elements[row][column] += self.elements[row][i] * rhs.elements[i][column];
                }
            }
        }
        return Matrix2D { elements };
    }
}

impl<const M: usize, const N: usize> Mul<f32> for Matrix2D<M, N> {
    type Output = Matrix2D<M, N>;

    fn mul(self, scalar: f32) -> Self::Output {
        let mut elements = self.elements.clone();
        for m in 0..M {
            for n in 0..N {
                elements[m][n] *= scalar;
            }
        }
        return Matrix2D { elements };
    }
}

#[cfg(test)]
mod tests {
    use crate::Matrix2D;

    #[test]
    fn add_test() {
        let expected: Matrix2D<2, 2> = Matrix2D::new([[3.0, 10.0], [10.0, -8.0]]);

        let a: Matrix2D<2, 2> = Matrix2D::new([[3.0, 7.0], [6.0, -4.0]]);
        let b: Matrix2D<2, 2> = Matrix2D::new([[0.0, 3.0], [4.0, -4.0]]);
        let actual: Matrix2D<2, 2> = a + b;

        // compilation error
        // let c: Matrix2D<2, 3> = Matrix2D::new([[0.0; 3], [0.0; 3]]);
        // a + c;

        assert_eq!(actual, expected);
    }

    #[test]
    fn add_assign_test() {
        let expected: Matrix2D<2, 2> = Matrix2D::new([[3.0, 10.0], [10.0, -8.0]]);

        let mut a: Matrix2D<2, 2> = Matrix2D::new([[3.0, 7.0], [6.0, -4.0]]);
        let b: Matrix2D<2, 2> = Matrix2D::new([[0.0, 3.0], [4.0, -4.0]]);
        a += b;

        // compilation error
        // let c: Matrix2D<2, 3> = Matrix2D::new([[0.0; 3], [0.0; 3]]);
        // a += c;

        assert_eq!(a, expected);
    }

    #[test]
    fn multiply_scalar_test() {
        let expected: Matrix2D<1, 3> = Matrix2D::new([[3.0, 6.0, 9.0]]);

        let a: Matrix2D<1, 3> = Matrix2D::new([[1.0, 2.0, 3.0]]);
        let actual: Matrix2D<1, 3> = a * 3.0;

        assert_eq!(actual, expected);
    }

    #[test]
    fn multiply_other_test() {
        let expected: Matrix2D<3, 3> =
            Matrix2D::new([[5.0, 11.0, 7.0], [5.0, 13.0, 16.0], [7.0, 17.0, 17.0]]);

        let a: Matrix2D<3, 2> = Matrix2D::new([[1.0, 2.0], [3.0, 1.0], [3.0, 2.0]]);
        let b: Matrix2D<2, 3> = Matrix2D::new([[1.0, 3.0, 5.0], [2.0, 4.0, 1.0]]);
        let actual: Matrix2D<3, 3> = a * b;

        // compilation error
        // let c: Matrix2D<1, 2> = Matrix2D::new([[3.0, 7.0]]);
        // a * c;

        assert_eq!(actual, expected);
    }
}
