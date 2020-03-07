export default class Helper {

    public static timeit(func: Function, floorIt = true): Promise<number> {
        return new Promise<number>(async (resolve) => {
            const startTime = Date.now()
            func()
            const duration = (Date.now() - startTime) / 1000.0
            if (floorIt) {
                resolve(Math.floor(duration))
            } else {
                resolve(duration)
            }
        })
    }
}