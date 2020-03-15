import Vector from "./vector";
import Tensor from "./tensor";
import Matrix from "./matrix";

export interface Example {
    data: Vector | Matrix | Tensor;
    label: Vector;
}
export default class Dataset {
    private data;
    BATCH_SIZE: number;
    IS_GENERATOR: boolean;
    TOTAL_EXAMPLES: number;
    DATA_STRUCTURE: any;
    GENERATOR: Function;
    size(): number;
    setGenerator(gen: Function): void;
    static read_image(path: string): Promise<Tensor>;
    vectorize_image(image: Tensor): Vector;
    loadMnistTrain(folderPath: string, maxExamples?: number, vectorize?: boolean): void;
    loadMnistTest(folderPath: string, maxExamples?: number, vectorize?: boolean): void;
    private loadMnist;
    loadTestData(path: string, maxExamples?: number): void;
    getBatch(batch: number): Array<Example>;
}
