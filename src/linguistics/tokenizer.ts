import {suffixes} from "./suffixes";
import ArrayHelper from "../lib/array_helper";
import fs from "fs"
import CsvParser from "./csv_parser";
import Dataset, {Example} from "../dataset";
import Tensor from "../tensor";

export default class Tokenizer {

    vocab: any = {}
    vocab_size = 0

    constructor() {}

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
        this.vocab_size = Object.keys(this.vocab).length
    }

    loadVocabulary(path: string) {
        this.vocab = JSON.parse(fs.readFileSync(path, {encoding: "utf-8"}))
    }

    saveVocabulary(path: string) {
        fs.writeFileSync(path, JSON.stringify(this.vocab))
    }

    tokenize(sentence: string): Tensor {
        return new Tensor(ArrayHelper.flatten(sentence.split(" ").map((word: string) => {
            const suffix = suffixes.filter((suff) => word.endsWith(suff.replace("-", "")))
            if (suffix.length > 0) {
                return [word.substr(0, word.lastIndexOf(suffix[0].replace("-", ""))), suffix]
            } else {
                return word
            }
        })).map((token: string) => Tensor.toCategorical(this.vocab[token], this.vocab_size).t))
    }

    createDataset(path: string, columns: number[], nr_labels): Dataset {
        const trainData = CsvParser.parse("./dataset/nlp/train.tsv", true)
        const dataset = new Dataset()

        const data: Example[] = CsvParser.filterColumns(trainData, columns).map((ex)  => {
            const label = Tensor.toCategorical(<number> ex[0], nr_labels)
            const data = this.tokenize(<string> ex[1])
            return {label: label, data: data}
        })

        const maxWordSize = data.reduce((acc, e) =>
            e.data.shape[0] > acc? e.data.shape[0]: acc, 0)

        data.forEach((ex: Example) => {
            const em = new Tensor([maxWordSize, this.vocab_size], true)
            ex.data.iterate((pos) => {
                em.set(pos, ex.data.get(pos))
            }, true)
            dataset.addExample({label: ex.label, data: em})
        })

        return dataset
    }
}