import Vector from "./vector";
import Matrix from "./matrix";

export default class Activations {

    private static sigmoidFunc = (x: number) => (1 / (1 + Math.exp(-x)))

    public static sigmoid(v: Vector | Matrix): Vector | Matrix {
        if (v instanceof Vector) {
            v.iterate((val: number, i: number) => {
                v.set(i, this.sigmoidFunc(val))
            });
        } else {
            v.iterate((i: number, j: number) => {
                v.set(i, j, this.sigmoidFunc(v.get(i,j)))
            });
        }

        return v
    }

    public static sigmoid_derivative(v: Vector | Matrix): Vector | Matrix {
        if (v instanceof Vector) {
            v.iterate((val: number, i: number) => {
                v.set(i, val * (1 - val))
            });
        } else {
            v.iterate((i: number, j: number) => {
                v.set(i, this.sigmoidFunc(v.get(i,j)) * (1 - this.sigmoidFunc(v.get(i,j))))
            });
        }

        return v
    }
}