"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const layer_1 = __importDefault(require("./layer"));
const tensor_1 = __importDefault(require("../tensor"));
class RecurrentLayer extends layer_1.default {
    constructor(hidden_size, vocab_size) {
        super();
        this.U = new tensor_1.default();
        this.V = new tensor_1.default();
        this.weights = new tensor_1.default();
        this.bias = new tensor_1.default();
        this.bias_hidden = new tensor_1.default();
        this.hidden_size = hidden_size;
        this.vocab_size = vocab_size;
    }
}
exports.default = RecurrentLayer;
