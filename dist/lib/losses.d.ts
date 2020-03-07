import Vector from "../vector";
import Matrix from "../matrix";
import {KernelFunction} from "gpu.js";

export default class Losses {
    static squared_error(m: Matrix, labels: Matrix): Matrix;
    static squared_error_derivative(m: Matrix, labels: Matrix): Matrix;
    static squared_error_derivative_gpu(): KernelFunction;
    static CrossEntropy(v: Vector, labels: Vector): Vector;
    static CrossEntropy_derivative(v: Vector, labels: Vector): Vector;
}
