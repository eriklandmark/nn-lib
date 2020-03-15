import Vector from "../../vector";
import Matrix from "../../matrix";
import {KernelFunction} from "gpu.js";
import CrossEntropy from "./cross_entropy";
import MeanSquaredError from "./mean_squared_error";

export interface ILoss {
    name: string
    normal(input: Matrix, labels: Matrix): Matrix
    derivative(input: Matrix, labels: Matrix): Matrix
    normal_gpu(): KernelFunction
    derivative_gpu(): KernelFunction
}

export default class Losses {

    static fromName(name: string) {
        switch (name) {
            case "cross_entropy": return new CrossEntropy()
            case "mean_squared_error": return new MeanSquaredError()
        }
    }

    public static CrossEntropy(v: Vector, labels: Vector): Vector {
        const out = new Vector(v.size());
        if (v.size() != labels.size()) throw "Labels and output vector doesn't match size..";
        out.iterate((_: number, i: number) => {
            const a = v.get(i);
            const y = labels.get(i)
            if (a == 1) {
                out.set(i, -Math.log(a))
            } else {
                out.set(i, -1*((y*Math.log(a)) + (1 - y)*Math.log(1 - a)))
            }
        });
        return out;
    }

    public static CrossEntropy_derivative(v: Vector, labels: Vector): Vector {
        const out = new Vector(v.size());
        if (v.size() != labels.size()) throw "Labels and output vector doesn't match size..";
        out.iterate((_: number, i: number) => {
            const a = v.get(i);
            const y = labels.get(i)
            out.set(i, (-y / a) + ((1 - y) / (1 - a)))
        });
        return out;
    }
}