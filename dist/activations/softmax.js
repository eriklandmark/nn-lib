"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const matrix_1 = __importDefault(require("../matrix"));
class Softmax {
    constructor() {
        this.name = "softmax";
    }
    normal(input) {
        const exp = input.exp();
        return exp.div(exp.sum(1, true));
    }
    derivative(input) {
        if (input instanceof matrix_1.default) {
            const m = input.copy(false);
            m.iterate((i, j) => {
                if (i == j) {
                    m.set(i, j, input.get(i, j) * (1 - input.get(i, j)));
                }
                else {
                    m.set(i, j, -(input.get(i, j) * input.get(i, j)));
                }
            });
            return m;
        }
        else {
            return input * (1 - input);
        }
    }
    normal_gpu() {
        return function actv(a) {
            let sum = 0;
            for (let i = 0; i < this.constants.softmax; i++) {
                sum += Math.exp(a[this.thread.y][i]);
            }
            return Math.exp(a[this.thread.y][this.thread.x]) / sum;
        };
    }
    derivative_gpu() {
        return function actv_der(a) {
            return a * (1 - a);
        };
    }
}
exports.default = Softmax;
