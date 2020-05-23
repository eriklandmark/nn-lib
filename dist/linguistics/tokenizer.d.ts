import Dataset from "../dataset";
export default class Tokenizer {
    vocab: any;
    vocab_size: number;
    constructor();
    createVocabulary(sentences: string[]): void;
    loadVocabulary(path: string): void;
    saveVocabulary(path: string): void;
    tokenize(sentence: string, normalize?: boolean): any[];
    createDataset(path: string, columns: number[]): Dataset;
}
