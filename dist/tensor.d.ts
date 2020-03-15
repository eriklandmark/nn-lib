import Vector from "./vector";

export default class Tensor {
    tensor: Float32Array[][];
    get: Function;
    set: Function;
    count: Function;
    dim(): {
        r: number;
        c: number;
        d: number;
    };
    shape(): number[];
    constructor(v?: number[][][] | Float32Array[][]);
    createEmptyArray(rows: number, columns: number, depth: number): void;
    static fromJsonObject(obj: any[][]): Tensor;
    iterate(func: Function): void;
    toString: (max_rows?: number) => string;
    copy(full?: boolean): Tensor;
    populateRandom(): void;
    empty(): boolean;
    vectorize(): Vector;
    div(val: number | Tensor): Tensor;
    mul(val: number | Tensor): Tensor;
    sub(val: number | Tensor): Tensor;
    add(val: number | Tensor): Tensor;
}
