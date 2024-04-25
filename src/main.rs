use type_safe_matrix::Matrix;

fn main() {
    let a: Matrix<2, 2> = Matrix::new([[3.0, 7.0], [6.0, -4.0]]);
    println!("{:?}", a);
}
