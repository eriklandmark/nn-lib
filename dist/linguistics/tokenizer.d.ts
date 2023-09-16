import Dataset from "../dataset";
import Tensor from "../tensor";
export default class Tokenizer {
    vocab: any;
    vocab_size: number;
    constructor();
    createVocabulary(sentences: string[]): void;
    loadVocabulary(path: string): void;
    saveVocabulary(path: string): void;
    tokenize(sentence: string): Tensor;
    createDataset(path: string, columns: number[], nr_labels: any): Dataset;
}
