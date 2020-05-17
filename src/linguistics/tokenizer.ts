import {suffixes} from "./suffixes";
import ArrayHelper from "../lib/array_helper";
import fs from "fs"
import CsvParser from "./csv_parser";
import Dataset, {Example} from "../dataset";
import Vector from "../vector";

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

    tokenize(sentence: string, normalize: boolean = false) {
        return ArrayHelper.flatten(sentence.split(" ").map((word: string) => {
            const suffix = suffixes.filter((suff) => word.endsWith(suff.replace("-", "")))
            if (suffix.length > 0) {
                return [word.substr(0, word.lastIndexOf(suffix[0].replace("-", ""))), suffix]
            } else {
                return word
            }
        })).map((token: string) => normalize? this.vocab[token] / this.vocab_size:this.vocab[token])
    }

    createDataset(path: string, columns: number[]): Dataset {
        const trainData = CsvParser.parse("./dataset/nlp/train.tsv", true)
        const dataset = new Dataset()

        const data: Example[] = CsvParser.filterColumns(trainData, columns).map((ex)  => {
            const label = Vector.toCategorical(<number> ex[0], 3)
            const data = new Vector(this.tokenize(<string> ex[1], true))
            return {label: label, data: data}
        })

        const maxVectorSize = data.reduce((acc, e) => (<Vector>e.data).size() > acc? (<Vector>e.data).size(): acc, 0)

        data.forEach((ex: Example) => {
            const em = new Vector(maxVectorSize)
            ex.data.iterate((val: number, i: number) => {
                em.set(i, val)
            })
            dataset.addExample({label: ex.label, data: em})
        })

        return dataset
    }
}