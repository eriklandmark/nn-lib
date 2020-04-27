"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const vector_1 = __importDefault(require("../vector"));
const cross_entropy_1 = __importDefault(require("./cross_entropy"));
const mean_squared_error_1 = __importDefault(require("./mean_squared_error"));
class Losses {
    static fromName(name) {
        switch (name) {
            case "cross_entropy": return new cross_entropy_1.default();
            case "mean_squared_error": return new mean_squared_error_1.default();
        }
    }
    static CrossEntropy(v, labels) {
        const out = new vector_1.default(v.size());
        if (v.size() != labels.size())
            throw "Labels and output vector doesn't match size..";
        out.iterate((_, i) => {
            const a = v.get(i);
            const y = labels.get(i);
            if (a == 1) {
                out.set(i, -Math.log(a));
            }
            else {
                out.set(i, -1 * ((y * Math.log(a)) + (1 - y) * Math.log(1 - a)));
            }
        });
        return out;
    }
    static CrossEntropy_derivative(v, labels) {
        const out = new vector_1.default(v.size());
        if (v.size() != labels.size())
            throw "Labels and output vector doesn't match size..";
        out.iterate((_, i) => {
            const a = v.get(i);
            const y = labels.get(i);
            out.set(i, (-y / a) + ((1 - y) / (1 - a)));
        });
        return out;
    }
}
exports.default = Losses;
