import Matrix from "../matrix";
import Vector from "../vector";
export default class MatrixHelper {
    static row_reduction(matrix: Matrix): Matrix;
    static linear_least_squares(x: Vector, y: Vector): void;
    static linear_least_squares_pol(x: Vector, y: Vector): void;
}
