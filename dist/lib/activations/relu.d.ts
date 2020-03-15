import Matrix from "../../matrix";
import IActivation from "./activations";
import {KernelFunction} from "gpu.js";

export default class ReLu implements IActivation {
    name: string;
    normal(input: Matrix): Matrix;
    derivative(input: Matrix): Matrix;
    normal_gpu(): KernelFunction;
    derivative_gpu(): KernelFunction;
}
