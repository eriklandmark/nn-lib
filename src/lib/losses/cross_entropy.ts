import ILoss from "./losses";
import Matrix from "../../matrix";
import {KernelFunction} from "gpu.js";

export default class CrossEntropy implements ILoss {

    name: string = "cross_entropy"
    epsilon: number = 10**-14

    normal(input: Matrix, labels: Matrix): Matrix {
        let out = input.copy(false)
        out.iterate((i: number, j: number) => {
            if (labels.get(i,j) != 0) {
                out.set(i, j, (labels.get(i,j) * Math.log(input.get(i,j) + this.epsilon)))
            }
        });
        return (<Matrix>(<Matrix> out).sum(1, true)).mul(-1)
    }

    derivative(input: Matrix, labels: Matrix): Matrix {
        return labels.mul(-1).div(input)
    }

    normal_gpu(): KernelFunction {
        return function actv(a, labels) {
            let sum = 0;
            for (let i = 0; i < this.constants.labels_length; i++) {
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