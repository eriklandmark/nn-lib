import Tensor from "../tensor";
export default class ReLu {
    constructor() {
        this.name = "relu";
    }
    normal(input) {
        if (input instanceof Tensor) {
            const m = input.copy(false);
            m.iterate((pos) => {
                m.set(pos, Math.max(input.get(pos), 0));
            }, true);
            return m;
        }
        else {
            return Math.max(input, 0);
        }
    }
    derivative(input) {
        if (input instanceof Tensor) {
            const m = input.copy(false);
            m.iterate((pos) => {
                m.set(pos, input.get(pos) > 0 ? 1 : 0);
            }, true);
            return m;
        }
        else {
            return input > 0 ? 1 : 0;
        }
    }
    normal_gpu() {
        return function actv(a) {
            return Math.max(a[this.thread.y][this.thread.x], 0);
        };
    }
    derivative_gpu() {
        return function actv(a) {
            return a > 0 ? 1 : 0;
        };
    }
}
