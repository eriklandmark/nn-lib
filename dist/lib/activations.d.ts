import Vector from "./vector";
import Matrix from "./matrix";
import {KernelFunction} from "gpu.js";

export interface IActivations {
    func: Function;
    derv: Function | null;
}
export default class Activations {
    static lookUp(name: string): IActivations;
    static sigmoid_gpu(): KernelFunction;
    static sigmoid(v: Vector | Matrix): Vector | Matrix;
    static sigmoid_derivative(v: Vector | Matrix): Vector | Matrix;
    static ReLu(v: Vector | Matrix): Vector | Matrix;
    static ReLu_derivative(v: Vector | Matrix): Vector | Matrix;
    static Softmax(m: Matrix): Matrix;
}
