"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const suffixes_1 = require("./suffixes");
const array_helper_1 = __importDefault(require("../helpers/array_helper"));
const fs_1 = __importDefault(require("fs"));
class Tokenizer {
    constructor() {
        this.vocab = {};
    }
    createVocabulary(sentences) {
        const sents = sentences.map((sentence) => sentence.trim().split(" "));
        const single_words = array_helper_1.default.delete_doublets(array_helper_1.default.flatten(sents));
        const vocab = array_helper_1.default.flatten(single_words.map((word) => {
            const suffix = suffixes_1.suffixes.filter((suff) => word.endsWith(suff.replace("-", "")));
            if (suffix.length > 0) {
                return [word.substr(0, word.lastIndexOf(suffix[0].replace("-", ""))), suffix];
            }
            else {
                return word;
            }
        }));
        this.vocab = vocab.sort().reduce((acc, token, index) => {
            acc[token.toString()] = index;
            return acc;
        }, {});
    }
    loadVocabulary(path) {
        this.vocab = JSON.parse(fs_1.default.readFileSync(path, { encoding: "utf-8" }));
    }
    saveVocabulary(path) {
        fs_1.default.writeFileSync(path, JSON.stringify(this.vocab));
    }
    tokenize(sentence) {
        return array_helper_1.default.flatten(sentence.split(" ").map((word) => {
            const suffix = suffixes_1.suffixes.filter((suff) => word.endsWith(suff.replace("-", "")));
            if (suffix.length > 0) {
                return [word.substr(0, word.lastIndexOf(suffix[0].replace("-", ""))), suffix];
            }
            else {
                return word;
            }
        })).map((token) => this.vocab[token]);
    }
}
exports.default = Tokenizer;
