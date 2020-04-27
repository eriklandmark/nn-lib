/*import fs from "fs"

import Tokenizer from "../src/linguistics/tokenizer";
import CsvParser from "../src/linguistics/csv_parser";
import ArrayHelper from "../src/helpers/array_helper";

const tokenizer = new Tokenizer()

const trainData = CsvParser.parse("./dataset/nlp/train.tsv", true)
const sentences = ArrayHelper.flatten(CsvParser.filterColumns(trainData, [3]))
tokenizer.createVocabulary(sentences)

console.log(tokenizer.tokenize(sentences[sentences.length-1]))*/

import Matrix from "../src/matrix";
import Helper from "../src/helpers/helper";

const size = 1000
const a = new Matrix()
a.createEmptyArray(size, size)
a.populateRandom()

const b = new Matrix()
b.createEmptyArray(size, size)
b.populateRandom()

Helper.timeit(() => {
    a.mm(b)
}, false).then((result) => {
    console.log(result)
})
