use std::ops::Add;
use std::ops::AddAssign;

#[derive(Debug, PartialEq)]
pub struct Matrix2D<const M: usize, const N: usize> {
    pub elements: [[f32; N]; M],
}

impl<const N: usize, const M: usize> Matrix2D<N, M> {
    pub fn new(elements: [[f32; M]; N]) -> Self {
        Matrix2D { elements }
    }
}

impl<const M: usize, const N: usize> Add for Matrix2D<M, N> {
    type Output = Matrix2D<M, N>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut elements = [[0.0; N]; M];
        for n in 0..M {
            for m in 0..N {
                elements[n][m] = self.elements[n][m] + rhs.elements[n][m];
            }
        }
        return Matrix2D { elements };
    }
}

impl<const M: usize, const N: usize> AddAssign for Matrix2D<M, N> {
    fn add_assign(&mut self, rhs: Self) {
        for n in 0..M {
            for m in 0..N {
                self.elements[n][m] += rhs.elements[n][m];
            }
        }
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
    fn add_add_assign() {
        let expected: Matrix2D<2, 2> = Matrix2D::new([[3.0, 10.0], [10.0, -8.0]]);

        let mut a: Matrix2D<2, 2> = Matrix2D::new([[3.0, 7.0], [6.0, -4.0]]);
        let b: Matrix2D<2, 2> = Matrix2D::new([[0.0, 3.0], [4.0, -4.0]]);
        a += b;

        // compilation error
        // let c: Matrix2D<2, 3> = Matrix2D::new([[0.0; 3], [0.0; 3]]);
        // a += c;

        assert_eq!(a, expected);
    }
}
