import ILoss from "./losses";
import {KernelFunction} from "gpu.js";
import Tensor from "../tensor";

export default class CrossEntropy implements ILoss {

    name: string = "cross_entropy"
    epsilon: number = 10**-14

    normal(input: Tensor, labels: Tensor): Tensor {
        let out = input.copy(false)
        out.iterate((pos) => {
            if (labels.get(pos) != 0) {
                out.set(pos, (labels.get(pos) * Math.log(input.get(pos) + this.epsilon)))
            }
        }, true);
        return (<Tensor>(<Tensor> out).sum(1, true)).mul(-1)
    }

    derivative(input: Tensor, labels: Tensor): Tensor {
        return labels.mul(-1).div(input)
    }

    normal_gpu(): KernelFunction {
        return function actv(a, labels) {
            let sum = 0;
            for (let i = 0; i < this.constants["labels_length"]; i++) {
                sum += labels[this.thread.y][i] * Math.log(a[this.thread.y][i] + 10**-14)
            }
             return sum * -1
        }
    }

    derivative_gpu(): KernelFunction {
        return function actv(a, labels) {
            labels
        }
    }
}