import Tensor from "../tensor";
export default class HyperbolicTangent {
    constructor() {
        this.name = "tanh";
    }
    normal_gpu() {
        return function actv(a) {
            return Math.tanh(a[this.thread.y][this.thread.x]);
        };
    }
    derivative_gpu() {
        return function actv_der(a) {
            return 1 - Math.pow(a, 2);
        };
    }
    normal(input) {
        if (input instanceof Tensor) {
            const m = input.copy(false);
            m.iterate((pos) => {
                m.set(pos, Math.tanh(input.get(pos)));
            }, true);
            return m;
        }
        else {
            return Math.tanh(input);
        }
    }
    derivative(input) {
        if (input instanceof Tensor) {
            const m = input.copy(false);
            m.iterate((pos) => {
                m.set(pos, 1 - Math.pow(input.get(pos), 2));
            }, true);
            return m;
        }
        else {
            return 1 - Math.pow(input, 2);
        }
    }
}
