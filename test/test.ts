/*

import Tokenizer from "../src/linguistics/tokenizer";
import CsvParser from "../src/linguistics/csv_parser";
import ArrayHelper from "../src/helpers/array_helper";

const tokenizer = new Tokenizer()

const trainData = CsvParser.parse("./dataset/nlp/train.tsv", true)
const sentences = ArrayHelper.flatten(CsvParser.filterColumns(trainData, [3]))
tokenizer.createVocabulary(sentences)

console.log(tokenizer.tokenize(sentences[sentences.length-1]))*/

import fs from "fs"
import Matrix from "../src/matrix";
import Vector from "../src/vector";
import {BacklogData} from "../src/model";

const data: BacklogData = JSON.parse(fs.readFileSync("./model/backlog.json", {encoding: "utf-8"}))

const batches = Object.keys(data.epochs).reduce((acc, epoch) => {
        acc.push(...data.epochs[epoch].batches.map((batch, index) => {
            batch["id"] = index
            batch["epoch_id"] = parseInt(epoch.substr(epoch.lastIndexOf("_") + 1));
            batch["global_id"] = batch["epoch_id"] - 1 == 0? index :
                (batch["epoch_id"] - 1) * data.epochs["epoch_" + (batch["epoch_id"] - 1)].batches.length + index
            return batch
        }))
        return acc
    }, [])


//const x = new Vector(Object.keys(data.epochs).slice(0, 100).map((epoch, index) => index + 1))
//const y = new Vector(Object.keys(data.epochs).slice(0, 100).map((epoch) => data.epochs[epoch].total_accuracy / 20))
const x = new Vector(batches.map((batch, index) => index + 1))
const y = new Vector(batches.map((batch) => batch.accuracy))


let A = new Matrix()
A.createEmptyArray(x.size(), 2)
A.matrix.forEach((val: Float32Array, index) => {
    A.set(index,0, 1)
    A.set(index,1, x.get(index))
})

let b = new Vector(y.size())
b.iterate((v, i) => {
    b.set(i, y.get(i))
})

const VL: Matrix = <Matrix> A.transpose().mm(A)
const HL: Vector = <Vector> A.transpose().mm(b)

let xV = VL.inv()!.mm(HL)

console.log(xV.toString())