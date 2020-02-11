import Vector from "./vector";
import validate = WebAssembly.validate;

export default class Activations {

    private static sigmoidFunc = (x: number) => (1 / (1 + Math.exp(-x)))

    public static sigmoid(v: Vector): Vector {
        v.iterate((val: number, i: number) => {
            v.set(i, this.sigmoidFunc(val))
        });
        return v
    }

    public static sigmoid_derivative(v: Vector): Vector {
        v.iterate((val: number, i: number) => {
            v.set(i, this.sigmoidFunc(val) * (1 - this.sigmoidFunc(val)))
        });
        return v
    }
}