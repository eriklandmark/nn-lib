export default class Tokenizer {
    vocab: any;
    constructor();
    createVocabulary(sentences: string[]): void;
    loadVocabulary(path: string): void;
    saveVocabulary(path: string): void;
    tokenize(sentence: string): any[];
}
