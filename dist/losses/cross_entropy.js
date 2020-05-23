"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class CrossEntropy {
    constructor() {
        this.name = "cross_entropy";
        this.epsilon = Math.pow(10, -14);
    }
    normal(input, labels) {
        let out = input.copy(false);
        out.iterate((pos) => {
            if (labels.get(pos) != 0) {
                out.set(pos, (labels.get(pos) * Math.log(input.get(pos) + this.epsilon)));
            }
        }, true);
        return out.sum(1, true).mul(-1);
    }
    derivative(input, labels) {
        return labels.mul(-1).div(input);
    }
    normal_gpu() {
        return function actv(a, labels) {
            let sum = 0;
            for (let i = 0; i < this.constants.labels_length; i++) {
                sum += labels[this.thread.y][i] * Math.log(a[this.thread.y][i] + Math.pow(10, -14));
            }
            return sum * -1;
        };
    }
    derivative_gpu() {
        return function actv(a, labels) {
            labels;
        };
    }
}
exports.default = CrossEntropy;
