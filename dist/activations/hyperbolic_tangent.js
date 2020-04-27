"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const matrix_1 = __importDefault(require("../matrix"));
class HyperbolicTangent {
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
        if (input instanceof matrix_1.default) {
            const m = input.copy(false);
            m.iterate((i, j) => {
                m.set(i, j, Math.tanh(input.get(i, j)));
            });
            return m;
        }
        else {
            return Math.tanh(input);
        }
    }
    derivative(input) {
        if (input instanceof matrix_1.default) {
            const m = input.copy(false);
            m.iterate((i, j) => {
                m.set(i, j, 1 - Math.pow(input.get(i, j), 2));
            });
            return m;
        }
        else {
            return 1 - Math.pow(input, 2);
        }
    }
}
exports.default = HyperbolicTangent;
