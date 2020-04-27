"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class CrossEntropy {
    constructor() {
        this.name = "cross_entropy";
        this.epsilon = Math.pow(10, -14);
    }
    normal(input, labels) {
        let out = input.copy(false);
        out.iterate((i, j) => {
            if (labels.get(i, j) != 0) {
                out.set(i, j, (labels.get(i, j) * Math.log(input.get(i, j) + this.epsilon)));
            }
        });
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
