import Matrix from "../../matrix";
import IActivation from "./activations";
import { GPUFunction, KernelFunction, ThreadKernelVariable } from "gpu.js";
export default class Softmax implements IActivation {
    name: string;
    normal(input: Matrix): Matrix;
    derivative(input: Matrix | number): Matrix | number;
    normal_gpu(): KernelFunction;
    derivative_gpu(): GPUFunction<ThreadKernelVariable[]>;
}
