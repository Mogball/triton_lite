from memory import UnsafePointer

@register_passable("trivial")
struct Ptr[T: AnyTrivialRegType, addrspace: UInt = 0]:
    alias _mlir_type = __mlir_type[
        `!kgen.pointer<`, T, `, `, addrspace.value, `>`
    ]

    var address: Self._mlir_type

    @always_inline("builtin")
    fn __init__(out self):
        self.address = __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: Self._mlir_type):
        self.address = value

    @always_inline
    fn offset[I: Indexer, //](self, idx: I) -> Self:
        return __mlir_op.`pop.offset`(self.address, index(idx))

    @always_inline
    fn __add__[I: Indexer, //](self, offset: I) -> Self:
        return self.offset(offset)

    @always_inline
    fn __sub__[I: Indexer, //](self, offset: I) -> Self:
        return self + (-1 * index(offset))

    @always_inline
    fn __iadd__[I: Indexer, //](mut self, offset: I):
        self = self + offset

    @always_inline
    fn __isub__[I: Indexer, //](mut self, offset: I):
        self = self - offset

    @always_inline
    fn store(self, value: T):
        alias alignment = UnsafePointer[T].alignment
        __mlir_op.`pop.store`[alignment = alignment.value](value, self.address)

    @always_inline
    fn load(self) -> T:
        alias alignment = UnsafePointer[T].alignment
        return __mlir_op.`pop.load`[alignment = alignment.value](self.address)
