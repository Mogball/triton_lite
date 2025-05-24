@register_passable("trivial")
struct Shape:
    var value: VariadicList[UInt]

    @implicit
    fn __init__[Intable: Intable](out self, value: Intable):
        self.value = VariadicList[UInt](UInt(Int(value)))

    @implicit
    fn __init__[I: Indexer](out self, values: (I,)):
        self.value = VariadicList[UInt](UInt(Int(values[0])))

    @implicit
    fn __init__[I0: Indexer, I1: Indexer](out self, values: (I0, I1)):
        self.value = VariadicList[UInt](
            UInt(Int(values[0])), UInt(Int(values[1]))
        )

    @implicit
    fn __init__[
        I0: Indexer, I1: Indexer, I2: Indexer
    ](out self, values: (I0, I1, I2)):
        self.value = VariadicList[UInt](
            UInt(Int(values[0])), UInt(Int(values[1])), UInt(Int(values[2]))
        )

    fn __getitem__(self, i: Int) -> UInt:
        return self.value[i]

    fn size(self) -> UInt:
        return len(self.value)

    fn product(self) -> UInt:
        res = 1
        for i in range(self.size()):
            val = self.value[i]
            res *= val
        return res
