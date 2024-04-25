# type-safe-matrix
A type-safe matrix math library.

## Example
```
use type_safe_matrix::Matrix;

fn main() {
    let a: Matrix<2, 2> = Matrix::new([[3.0, 7.0], [6.0, -4.0]]);
    let b: Matrix<2, 2> = Matrix::new([[0.0, 3.0], [4.0, -4.0]]);
    let c: Matrix<2, 2> = a + b;

    // compilation error, because 2x3 matrix can not be added to 2x2 matrix.
    // let d: Matrix<2, 3> = Matrix::new([[0.0; 3], [0.0; 3]]);
    // let e = a + d;

    let x: Matrix<3, 2> = Matrix::new([[1.0, 2.0], [3.0, 1.0], [3.0, 2.0]]);
    let y: Matrix<2, 3> = Matrix::new([[1.0, 3.0, 5.0], [2.0, 4.0, 1.0]]);
    let z: Matrix<3, 3> = x * y;

    // compilation error, because 1x2 matrix can not be multiplied by 3x2 matrix.
    // let w: Matrix<1, 2> = Matrix::new([[3.0, 7.0]]);
    // let q = x * w;
}
```
