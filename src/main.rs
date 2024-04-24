use type_safe_matrix::Matrix2D;

fn main() {
    let a: Matrix2D<2, 2> = Matrix2D::new([[3.0, 7.0], [6.0, -4.0]]);
    println!("{:?}", a);
}
