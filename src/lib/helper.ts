export default class Helper {

    public static timeit(func: Function, floorIt: boolean = true): Promise<number> {
        return new Promise<number>(async (resolve) => {
            const startTime = Date.now()
            await func()
            const duration = (Date.now() - startTime) / 1000.0
            if (floorIt) {
                resolve(Math.floor(duration))
            } else {
                resolve(duration)
            }
        })
    }

    public static timeitSync(func: Function, floorIt: boolean = true): number {
        const startTime = Date.now()
        func()
        const duration = (Date.now() - startTime) / 1000.0
        if (floorIt) {
            return Math.floor(duration)
        } else {
            return duration
        }
    }
}