import Vector from "./vector";

export interface Example {
    data: Vector;
    label: Vector;
}
export default class Dataset {
    private data;
    BATCH_SIZE: number;
    IS_GENERATOR: boolean;
    TOTAL_EXAMPLES: number;
    GENERATOR: Function;
    size(): number;
    setGenerator(gen: Function): void;
    static read_image(path: string): Promise<Vector>;
    loadMnistTrain(folderPath: string, maxExamples?: number): void;
    loadMnistTest(folderPath: string, maxExamples?: number): void;
    private loadMnist;
    loadTestData(path: string, maxExamples?: number): void;
    getBatch(batch: number): Array<Example>;
}
