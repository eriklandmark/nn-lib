"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const suffixes_1 = require("./suffixes");
const array_helper_1 = __importDefault(require("../lib/array_helper"));
const fs_1 = __importDefault(require("fs"));
const csv_parser_1 = __importDefault(require("./csv_parser"));
const dataset_1 = __importDefault(require("../dataset"));
const tensor_1 = __importDefault(require("../tensor"));
class Tokenizer {
    constructor() {
        this.vocab = {};
        this.vocab_size = 0;
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
        this.vocab_size = Object.keys(this.vocab).length;
    }
    loadVocabulary(path) {
        this.vocab = JSON.parse(fs_1.default.readFileSync(path, { encoding: "utf-8" }));
    }
    saveVocabulary(path) {
        fs_1.default.writeFileSync(path, JSON.stringify(this.vocab));
    }
    tokenize(sentence, normalize = false) {
        return array_helper_1.default.flatten(sentence.split(" ").map((word) => {
            const suffix = suffixes_1.suffixes.filter((suff) => word.endsWith(suff.replace("-", "")));
            if (suffix.length > 0) {
                return [word.substr(0, word.lastIndexOf(suffix[0].replace("-", ""))), suffix];
            }
            else {
                return word;
            }
        })).map((token) => normalize ? this.vocab[token] / this.vocab_size : this.vocab[token]);
    }
    createDataset(path, columns) {
        const trainData = csv_parser_1.default.parse("./dataset/nlp/train.tsv", true);
        const dataset = new dataset_1.default();
        const data = csv_parser_1.default.filterColumns(trainData, columns).map((ex) => {
            const label = tensor_1.default.toCategorical(ex[0], 3);
            const data = new tensor_1.default(this.tokenize(ex[1], true));
            return { label: label, data: data };
        });
        const maxVectorSize = data.reduce((acc, e) => e.data.count() > acc ? e.data.count() : acc, 0);
        data.forEach((ex) => {
            const em = new tensor_1.default([maxVectorSize], true);
            ex.data.iterate((pos) => {
                em.set(pos, ex.data.get(pos));
            }, true);
            dataset.addExample({ label: ex.label, data: em });
        });
        return dataset;
    }
}
exports.default = Tokenizer;
