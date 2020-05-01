/*import fs from "fs"

import Tokenizer from "../src/linguistics/tokenizer";
import CsvParser from "../src/linguistics/csv_parser";
import ArrayHelper from "../src/helpers/array_helper";

const tokenizer = new Tokenizer()

const trainData = CsvParser.parse("./dataset/nlp/train.tsv", true)
const sentences = ArrayHelper.flatten(CsvParser.filterColumns(trainData, [3]))
tokenizer.createVocabulary(sentences)

console.log(tokenizer.tokenize(sentences[sentences.length-1]))*/

let b_id = 0;
let id = 0;

for (let i = 0; i < 20; i++) {
    for (let j = 0; j < 10; j++) {
       if(i == 0) {
           console.log(j)
       } else {
           console.log(i * 10 + j)
       }
    }
}