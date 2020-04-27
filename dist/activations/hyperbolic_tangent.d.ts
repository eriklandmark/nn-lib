import { IActivation } from "./activations";
import Matrix from "../matrix";
import { GPUFunction, KernelFunction, ThreadKernelVariable } from "gpu.js";
export default class HyperbolicTangent implements IActivation {
    name: string;
    normal_gpu(): KernelFunction;
    derivative_gpu(): GPUFunction<ThreadKernelVariable[]>;
    normal(input: Matrix | number): Matrix | number;
    derivative(input: Matrix | number): Matrix | number;
}
