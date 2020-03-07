import Vector from "../vector";
import Matrix from "../matrix";
import {KernelFunction} from "gpu.js";

export default class Losses {

    public static squared_error(m: Matrix, labels: Matrix): Matrix {
        if (m.dim().r != labels.dim().r || m.dim().c != labels.dim().c) throw "Labels and output vector doesn't match size..";
        return m.sub(labels).pow(2)
    }

    public static squared_error_derivative(m: Matrix, labels: Matrix): Matrix {
        if (m.dim().r != labels.dim().r || m.dim().c != labels.dim().c) throw "Labels and output vector doesn't match size..";
        return m.sub(labels)
    }

    public static squared_error_derivative_gpu(): KernelFunction {
        return function loss(m, label) {
            //@ts-ignore
            return m - label
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