import { IActivation } from "../activations/activations";
import { ILoss } from "./losses";
import Matrix from "../matrix";
export default class Gradients {
    static get_gradient(actvFunc: IActivation, lossFunc: ILoss): IGradient;
}
export interface IGradient {
    (input: Matrix, labels: Matrix): Matrix;
}
