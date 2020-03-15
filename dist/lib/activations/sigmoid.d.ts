import IActivation from "./activations";
import {KernelFunction} from "gpu.js";
import Matrix from "../../matrix";

export default class Sigmoid implements IActivation {
    name: string;
    normal_gpu(): KernelFunction;
    derivative_gpu(): KernelFunction;
    normal(input: Matrix | number): Matrix | number;
    derivative(input: Matrix | number): Matrix | number;
}
