import fs from "fs"

import Tokenizer from "../src/linguistics/tokenizer";
import CsvParser from "../src/linguistics/csv_parser";
import ArrayHelper from "../src/helpers/array_helper";

const tokenizer = new Tokenizer()

const trainData = CsvParser.parse("./dataset/nlp/train.tsv", true)
const sentences = ArrayHelper.flatten(CsvParser.filterColumns(trainData, [3]))
tokenizer.createVocabulary(sentences)

console.log(tokenizer.tokenize(sentences[sentences.length-1]))