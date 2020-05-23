export default class Helper {
    static timeit(func: Function, floorIt?: boolean): Promise<number>;
    static timeitSync(func: Function, floorIt?: boolean): number;
}
