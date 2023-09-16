import Tensor from "../tensor";
export default class Sigmoid {
    constructor() {
        this.name = "sigmoid";
    }
    normal_gpu() {
        return function actv(a) {
            return (1 / (1 + Math.exp(-a[this.thread.y][this.thread.x])));
        };
    }
    derivative_gpu() {
        return function actv_der(a) {
            return a * (1 - a);
        };
    }
    normal(input) {
        if (input instanceof Tensor) {
            const m = input.copy(false);
            m.iterate((pos) => {
                m.set(pos, 1 / (1 + Math.exp(-input.get(pos))));
            }, true);
            return m;
        }
        else {
            return 1 / (1 + Math.exp(-input));
        }
    }
    derivative(input) {
        if (input instanceof Tensor) {
            const m = input.copy(false);
            m.iterate((pos) => {
                m.set(pos, input.get(pos) * (1 - input.get(pos)));
            }, true);
            return m;
        }
        else {
            return input * (1 - input);
        }
    }
}
