import {IActivation} from "../activations/activations";
import {ILoss} from "./losses";
import Matrix from "../../matrix";

export default class Gradients {
    public static get_gradient(actvFunc: IActivation, lossFunc: ILoss): IGradient {
        let gradientFunc: IGradient
        if (actvFunc.name == "softmax" && lossFunc.name == "cross_entropy") {
            gradientFunc = function (input, labels) {
                return input.sub(labels)
            }
        } else if (actvFunc.name == "sigmoid" && lossFunc.name == "mean_squared_error") {
            gradientFunc = function (input, labels) {
                return input.sub(labels)
            }
        }

        return gradientFunc
    }
}

export interface IGradient {
    (input: Matrix, labels: Matrix): Matrix
}