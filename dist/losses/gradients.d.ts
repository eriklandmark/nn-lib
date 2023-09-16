import { IActivation } from "../activations/activations";
import { ILoss } from "./losses";
import Tensor from "../tensor";
export default class Gradients {
    static get_gradient(actvFunc: IActivation, lossFunc: ILoss): IGradient;
}
export interface IGradient {
    (input: Tensor, labels: Tensor): Tensor;
}
