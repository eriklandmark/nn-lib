import Vector from "../../vector";
import Matrix from "../../matrix";
import { KernelFunction } from "gpu.js";
import CrossEntropy from "./cross_entropy";
import MeanSquaredError from "./mean_squared_error";
export interface ILoss {
    name: string;
    normal(input: Matrix, labels: Matrix): Matrix;
    derivative(input: Matrix, labels: Matrix): Matrix;
    normal_gpu(): KernelFunction;
    derivative_gpu(): KernelFunction;
}
export default class Losses {
    static fromName(name: string): CrossEntropy | MeanSquaredError;
    static CrossEntropy(v: Vector, labels: Vector): Vector;
    static CrossEntropy_derivative(v: Vector, labels: Vector): Vector;
}
