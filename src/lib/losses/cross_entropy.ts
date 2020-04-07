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
                //+((1 - labels.get(i,j))*Math.log10(1 - input.get(i,j) + this.epsilon)))
            }
        });
        return (<Matrix>(<Matrix> out).sum(1, true)).mul(-1)
    }

    derivative(input: Matrix, labels: Matrix): Matrix {
        return labels.mul(-1).div(input)
    }

    normal_gpu(): KernelFunction {
        return function actv() {}
    }

    derivative_gpu(): KernelFunction {
        return function actv() {}
    }
}