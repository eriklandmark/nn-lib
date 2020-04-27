"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const matrix_1 = __importDefault(require("../matrix"));
class ReLu {
    constructor() {
        this.name = "relu";
    }
    normal(input) {
        if (input instanceof matrix_1.default) {
            const m = input.copy(false);
            m.iterate((i, j) => {
                m.set(i, j, Math.max(input.get(i, j), 0));
            });
            return m;
        }
        else {
            return Math.max(input, 0);
        }
    }
    derivative(input) {
        if (input instanceof matrix_1.default) {
            const m = input.copy(false);
            m.iterate((i, j) => {
                m.set(i, j, input.get(i, j) > 0 ? 1 : 0);
            });
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
exports.default = ReLu;
