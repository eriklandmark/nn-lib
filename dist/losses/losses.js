"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const cross_entropy_1 = __importDefault(require("./cross_entropy"));
const mean_squared_error_1 = __importDefault(require("./mean_squared_error"));
class Losses {
    static fromName(name) {
        switch (name) {
            case "cross_entropy": return cross_entropy_1.default;
            case "mean_squared_error": return mean_squared_error_1.default;
        }
    }
}
exports.default = Losses;
