import Vector from "./vector";
import Matrix from "./matrix";
import {KernelFunction} from "gpu.js";

export default class Activations {

    private static sigmoidFunc = (x: number) => (1 / (1 + Math.exp(-x)))

    public static sigmoid_gpu(): KernelFunction {
        return function actv(a) {
            return (1 / (1 + Math.exp(-a)))
        }
    }

    public static sigmoid(v: Vector | Matrix): Vector | Matrix {
        if (v instanceof Vector) {
            v.iterate((val: number, i: number) => {
                v.set(i, (1 / (1 + Math.exp(-val))))
            });
            return v
        } else {
            const m = v.copy()
            m.iterate((i: number, j: number) => {
                m.set(i, j, 1 / (1 + Math.exp(-v.get(i,j))))
            });
            return m
        }
    }

    public static sigmoid_derivative(v: Vector | Matrix): Vector | Matrix {
        if (v instanceof Vector) {
            v.iterate((val: number, i: number) => {
                v.set(i, val * (1 - val))
            });
            return v;
        } else {
            const m = v.copy()
            m.iterate((i: number, j: number) => {
                m.set(i, j, m.get(i,j) * (1 - m.get(i,j)))
            });
            return m
        }
    }

    public static ReLu(v: Vector) : Vector {
        console.log(v.toString())
        v.iterate((val: number, i: number) => {
            v.set(i, Math.max(val, 0))
        });
        console.log(v.toString())
        return v
    }

    public static ReLu_derivative(v: Vector) : Vector {
        v.iterate((val: number, i: number) => {
            v.set(i, val > 0 ? 1 : 0)
        });

        return v
    }

    public static Softmax(m: Matrix): Matrix {
        const exp = m.exp();
        return exp.div(exp.sum(1, true))
    }
}