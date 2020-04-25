import Vector from "./vector";
import Matrix from "./matrix";
export default class Tensor {
    tensor: Float64Array[][];
    get: Function;
    set: Function;
    count: Function;
    dim(): {
        r: number;
        c: number;
        d: number;
    };
    shape(): number[];
    constructor(v?: number[][][] | Float64Array[][]);
    createEmptyArray(rows: number, columns: number, depth: number): void;
    static fromJsonObject(obj: any[][]): Tensor;
    toNumberArray(): number[][];
    iterate(func: Function, channel_first?: boolean): void;
    toString: (max_rows?: number) => string;
    copy(full?: boolean): Tensor;
    populateRandom(): void;
    empty(): boolean;
    vectorize(channel_first?: boolean): Vector;
    div(val: number | Tensor): Tensor;
    mul(val: number | Tensor): Tensor;
    sub(val: number | Tensor): Tensor;
    add(val: number | Tensor): Tensor;
    padding(padding_height: any, padding_width: any): Tensor;
    im2patches(patch_height: number, patch_width: number, filter_height: number, filter_width: number): Matrix;
    rotate180(): Tensor;
}
