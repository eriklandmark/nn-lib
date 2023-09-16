import Tensor from "./tensor";
export interface Example {
    data: Tensor;
    label: Tensor;
}
export default class Dataset {
    private data;
    VERBOSE: boolean;
    BATCH_SIZE: number;
    IS_GENERATOR: boolean;
    TOTAL_EXAMPLES: number;
    DATA_SHAPE: number[];
    GENERATOR: Function;
    size(): number;
    setGenerator(gen: Function): void;
    addExample(ex: Example): void;
    static read_image(path: string, channels?: number): Promise<Tensor>;
    loadMnistTrain(folderPath: string, maxExamples?: number, vectorize?: boolean): void;
    loadMnistTest(folderPath: string, maxExamples?: number, vectorize?: boolean): void;
    shuffle(): void;
    private loadMnist;
    loadTestData(path: string, maxExamples?: number): void;
    getBatch(batch_id: number): Example[];
}
