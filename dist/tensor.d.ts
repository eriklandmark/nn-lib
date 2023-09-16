export default class Tensor {
    t: Float64Array | Float64Array[] | Float64Array[][] | Float64Array[][][];
    shape: number[];
    dim: number;
    constructor(v?: any[] | Float64Array, shape?: boolean);
    static createIdentityMatrix(n: number): Tensor;
    private createTensor;
    private createFromShape;
    private calculateShape;
    get(pos: number[], to?: number[]): any;
    getLinearPos(pos: number[]): number;
    set(pos: number[], v: number): void;
    count(): number;
    static toCategorical(index: number, size: number): Tensor;
    static fromJsonObject(obj: any[][]): Tensor;
    equalShape(t: Tensor): boolean;
    equal(t: Tensor, err?: number): boolean;
    toNumberArray(): any[];
    iterate(func: Function, use_pos?: boolean, channel_first?: boolean): void;
    numberToString(nr: number, precision?: number, autoFill?: boolean): string;
    toString(max_rows?: number, precision?: number): string;
    print(max_rows?: number, precision?: number): void;
    copy(full?: boolean): Tensor;
    populateRandom(seed?: number | null): void;
    empty(): boolean;
    vectorize(channel_first?: boolean): Tensor;
    div(v: number | Tensor, safe?: boolean): Tensor;
    mul(v: number | Tensor): Tensor;
    sub(v: number | Tensor): Tensor;
    add(v: number | Tensor): Tensor;
    pow(v: number): Tensor;
    sqrt(): Tensor;
    exp(base?: null | number): Tensor;
    inv_el(eps?: number): Tensor;
    fill(scalar: number): Tensor;
    log(): Tensor;
    dot(b: Tensor): Tensor;
    padding(padding_height: number, padding_width: number, axis?: number[]): Tensor;
    im2patches(patch_height: number, patch_width: number, filter_height: number, filter_width: number): Tensor;
    rotate180(): Tensor;
    rowVectors(): Tensor[];
    argmax(index?: number, axis?: number): number;
    reshape(shape: number[]): Tensor;
    sum(axis?: number, keepDims?: boolean): number | Tensor;
    norm(p?: number): number;
    mean(axis?: number, keep_dims?: boolean): number | Tensor;
    repeat(axis?: number, times?: number): Tensor;
    transpose(): Tensor;
    trace(): number;
    isUpperTriangular(): boolean;
    isLowerTriangular(): boolean;
    isSymmetric(): boolean;
    isDiagonal(): boolean;
    rref(verify?: boolean): Tensor;
    swapRows(i: any, j: any): void;
    inv(): Tensor;
    extend(b: Tensor, axis?: number): Tensor;
    lu(): Tensor[];
    solve(b: Tensor): Tensor;
    solve2(b: Tensor): Tensor;
    det(): number;
    cond(): number;
    concatenate(t: Tensor, direction: "h" | "v"): Tensor;
}
