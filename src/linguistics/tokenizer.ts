import {suffixes} from "./suffixes";
import ArrayHelper from "../helpers/array_helper";
import fs from "fs"

export default class Tokenizer {

    vocab: any = {}

    constructor() {

    }

    createVocabulary(sentences: string[]) {
        const sents = sentences.map((sentence) => sentence.trim().split(" "))
        const single_words = ArrayHelper.delete_doublets(ArrayHelper.flatten(sents))
        const vocab = ArrayHelper.flatten(single_words.map((word: string) => {
            const suffix = suffixes.filter((suff) => word.endsWith(suff.replace("-", "")))
            if (suffix.length > 0) {
                return [word.substr(0, word.lastIndexOf(suffix[0].replace("-", ""))), suffix]
            } else {
                return word
            }
        }))
        this.vocab = vocab.sort().reduce((acc, token: string, index: number) => {
            acc[token.toString()] = index
            return acc
        }, {})
    }

    loadVocabulary(path: string) {
        this.vocab = JSON.parse(fs.readFileSync(path, {encoding: "utf-8"}))
    }

    saveVocabulary(path: string) {
        fs.writeFileSync(path, JSON.stringify(this.vocab))
    }

    tokenize(sentence: string) {
        return ArrayHelper.flatten(sentence.split(" ").map((word: string) => {
            const suffix = suffixes.filter((suff) => word.endsWith(suff.replace("-", "")))
            if (suffix.length > 0) {
                return [word.substr(0, word.lastIndexOf(suffix[0].replace("-", ""))), suffix]
            } else {
                return word
            }
        })).map((token: string) => this.vocab[token])
    }
}