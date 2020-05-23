"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const tensor_1 = __importDefault(require("../tensor"));
class Elu {
    constructor() {
        this.name = "elu";
        this.a = 1;
    }
    normal(input) {
        if (input instanceof tensor_1.default) {
            const m = input.copy(false);
            m.iterate((pos) => {
                m.set(pos, input.get(pos) > 0 ? input.get(pos) : this.a * (Math.exp(input.get(pos)) - 1));
            }, true);
            return m;
        }
        else {
            return input > 0 ? input : this.a * (Math.exp(input) - 1);
        }
    }
    derivative(input) {
        if (input instanceof tensor_1.default) {
            const m = input.copy(false);
            m.iterate((pos) => {
                m.set(pos, input.get(pos) > 0 ? 1 : input.get(pos) + this.a);
            }, true);
            return m;
        }
        else {
            return input > 0 ? 1 : input + this.a;
        }
    }
    normal_gpu() {
        return function actv(a) {
            return a > 0 ? a : Math.exp(a) - 1;
        };
    }
    derivative_gpu() {
        return function actv(a) {
            return a > 0 ? 1 : a + 1;
        };
    }
}
exports.default = Elu;
