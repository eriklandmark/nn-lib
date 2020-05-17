import Tokenizer from "../src/linguistics/tokenizer";
import CsvParser from "../src/linguistics/csv_parser";
import ArrayHelper from "../src/lib/array_helper";

const tokenizer = new Tokenizer()

const trainData = CsvParser.parse("./dataset/nlp/train.tsv", true)
const sentences = ArrayHelper.flatten(CsvParser.filterColumns(trainData, [3]))
tokenizer.createVocabulary(sentences)

const dataset = tokenizer.createDataset("./dataset/nlp/train.tsv", [1,3])

console.log(dataset.size())
dataset.BATCH_SIZE = dataset.size()
console.log(dataset.getBatch(0)[0].data.toString())

