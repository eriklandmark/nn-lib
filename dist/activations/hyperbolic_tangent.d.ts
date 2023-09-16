import { IActivation } from "./activations";
import { GPUFunction, KernelFunction, ThreadKernelVariable } from "gpu.js";
import Tensor from "../tensor";
export default class HyperbolicTangent implements IActivation {
    name: string;
    normal_gpu(): KernelFunction;
    derivative_gpu(): GPUFunction<ThreadKernelVariable[]>;
    normal(input: Tensor | number): Tensor | number;
    derivative(input: Tensor | number): Tensor | number;
}
