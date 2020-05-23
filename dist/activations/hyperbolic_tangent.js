"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const tensor_1 = __importDefault(require("../tensor"));
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
        if (input instanceof tensor_1.default) {
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
        if (input instanceof tensor_1.default) {
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
exports.default = HyperbolicTangent;
