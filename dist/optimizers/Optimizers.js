"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const StochasticGradientDescent_1 = __importDefault(require("./StochasticGradientDescent"));
const Adam_1 = __importDefault(require("./Adam"));
class Optimizers {
    static fromName(name) {
        switch (name) {
            case "sgd": return StochasticGradientDescent_1.default;
            case "adam": return Adam_1.default;
        }
    }
}
exports.default = Optimizers;
