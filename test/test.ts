import Tensor from "../src/tensor";
import Helper from "../src/lib/helper";

const size = 1000;

const a = new Tensor([size, size], true)
a.populateRandom()

const b = new Tensor([size, size], true)

b.populateRandom()

const t = Helper.timeitSync(() => {
    a.dot(b)
}, false)

console.log(t)


/*import Tokenizer from "../src/linguistics/tokenizer";
import CsvParser from "../src/linguistics/csv_parser";
import ArrayHelper from "../src/lib/array_helper";
import Tensor from "../src/tensor";


Tensor.toCategorical(1,2).print()

const tokenizer = new Tokenizer()
const trainData = CsvParser.parse("./dataset/nlp/train.tsv", true)
tokenizer.createVocabulary(<string[]> ArrayHelper.flatten(CsvParser.filterColumns(trainData, [3])))

console.log(tokenizer.vocab_size)
const ds = tokenizer.createDataset("./dataset/nlp/train.tsv", [1,3], 3)

ds.getBatch(0)[0].data.print()
*/