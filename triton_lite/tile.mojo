from .ptr import Ptr
from .shape import Shape
from .layout import Layout

from utils.static_tuple import StaticTuple
from gpu import thread_idx
from math import ceildiv
from sys._assembly import inlined_assembly
from collections import OptionalReg


fn _product(x: List[UInt]) -> UInt:
    res = 1
    for i in range(len(x)):
        res *= x[i]
    return res


fn _layout_nelts[T: Layout](layout: T, shape: Shape) -> UInt:
    return _product(layout.get_total_size_per_thread(shape))


@value
@register_passable
struct Tile[
    LayoutType: Layout, //,
    shape: Shape,
    T: AnyTrivialRegType,
    layout: LayoutType,
]:
    alias nelts: UInt = _layout_nelts(layout, shape)
    var impl: StaticTuple[T, Self.nelts]

    @implicit
    @always_inline
    fn __init__(out self: Tile[shape, UInt32, layout], broadcast: UInt):
        self.impl = StaticTuple[UInt32, Self.nelts]()

        @parameter
        for i in range(Int(Self.nelts)):
            self.impl[i] = UInt32(broadcast)

    @implicit
    @always_inline
    fn __init__(out self: Tile[shape, UInt32, layout], broadcast: Int):
        self.impl = StaticTuple[UInt32, Self.nelts]()

        @parameter
        for i in range(Int(Self.nelts)):
            self.impl[i] = UInt32(broadcast)

    @implicit
    @always_inline
    fn __init__[
        dt: DType
    ](out self: Tile[shape, Scalar[dt], layout], broadcast: FloatLiteral):
        self.impl = StaticTuple[Scalar[dt], Self.nelts]()

        @parameter
        for i in range(Int(Self.nelts)):
            self.impl[i] = Scalar[dt](broadcast)

    @implicit
    @always_inline
    fn __init__(out self, broadcast: T):
        self.impl = StaticTuple[T, Self.nelts]()

        @parameter
        for i in range(Int(Self.nelts)):
            self.impl[i] = broadcast

    @always_inline
    fn __init__(out self):
        self.impl = StaticTuple[T, Self.nelts]()

    @always_inline
    fn __add__[
        T: AnyTrivialRegType, addrspace: UInt
    ](
        self: Tile[shape, Ptr[T, addrspace], layout],
        rhs: Tile[shape, UInt32, layout],
    ) -> __type_of(self):
        res = __type_of(self)()

        @parameter
        for i in range(Int(Self.nelts)):
            res.impl[i] = self.impl[i] + rhs.impl[i]

        return res

    @always_inline
    fn __radd__(
        self: Tile[shape, UInt32, layout], lhs: Ptr[*_, **_]
    ) -> Tile[shape, __type_of(lhs), layout,]:
        return Tile[shape, __type_of(lhs), layout](broadcast=lhs) + self

    @always_inline
    fn __radd__(
        self: Tile[shape, UInt32, layout], lhs: Int
    ) -> Tile[shape, UInt32, layout]:
        return Tile[shape, UInt32, layout](broadcast=lhs) + self

    @always_inline
    fn elementwise_unary[
        dt: DType, f: fn (arg: Scalar[dt]) -> Scalar[dt]
    ](self: Tile[shape, Scalar[dt], layout]) -> Tile[shape, Scalar[dt], layout]:
        res = Tile[shape, Scalar[dt], layout]()

        @parameter
        for i in range(Int(Self.nelts)):
            res.impl[i] = f(self.impl[i])

        return res

    @always_inline
    fn elementwise_binary[
        dt: DType, f: fn (lhs: Scalar[dt], rhs: Scalar[dt]) -> Scalar[dt]
    ](
        self: Tile[shape, Scalar[dt], layout],
        other: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[dt], layout]:
        res = Tile[shape, Scalar[dt], layout]()

        @parameter
        for i in range(Int(Self.nelts)):
            res.impl[i] = f(self.impl[i], other.impl[i])

        return res

    @always_inline
    fn elementwise_comparison[
        dt: DType,
        f: fn (lhs: Scalar[dt], rhs: Scalar[dt]) -> Scalar[DType.bool],
    ](
        self: Tile[shape, Scalar[dt], layout],
        other: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[DType.bool], layout]:
        res = Tile[shape, Scalar[DType.bool], layout]()

        @parameter
        for i in range(Int(Self.nelts)):
            res.impl[i] = f(self.impl[i], other.impl[i])

        return res

    @always_inline
    fn __neg__[
        dt: DType
    ](self: Tile[shape, Scalar[dt], layout]) -> Tile[shape, Scalar[dt], layout]:
        return self.elementwise_unary[dt, Scalar[dt].__neg__]()

    @always_inline
    fn __add__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        other: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[dt], layout]:
        return self.elementwise_binary[dt, Scalar[dt].__add__](other)

    @always_inline
    fn __radd__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        other: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[dt], layout]:
        return self + other

    @always_inline
    fn __sub__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        other: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[dt], layout]:
        return self.elementwise_binary[dt, Scalar[dt].__sub__](other)

    @always_inline
    fn __mul__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        other: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[dt], layout]:
        return self.elementwise_binary[dt, Scalar[dt].__mul__](other)

    @always_inline
    fn __rmul__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        other: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[dt], layout]:
        return self * other

    @always_inline
    fn __rsub__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        other: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[dt], layout]:
        return other - self

    @always_inline
    fn __mod__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        rhs: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[dt], layout]:
        return self.elementwise_binary[dt, Scalar[dt].__mod__](rhs)

    @always_inline
    fn __floordiv__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        rhs: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[dt], layout]:
        return self.elementwise_binary[dt, Scalar[dt].__floordiv__](rhs)

    @always_inline
    fn __and__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        rhs: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[dt], layout]:
        return self.elementwise_binary[dt, Scalar[dt].__and__](rhs)

    @always_inline
    fn __or__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        rhs: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[dt], layout]:
        return self.elementwise_binary[dt, Scalar[dt].__or__](rhs)

    @always_inline
    fn __lt__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        rhs: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[DType.bool], layout]:
        return self.elementwise_comparison[dt, Scalar[dt].__lt__](rhs)

    @always_inline
    fn __le__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        rhs: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[DType.bool], layout]:
        return self.elementwise_comparison[dt, Scalar[dt].__le__](rhs)

    @always_inline
    fn __gt__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        rhs: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[DType.bool], layout]:
        return self.elementwise_comparison[dt, Scalar[dt].__gt__](rhs)

    @always_inline
    fn __ge__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        rhs: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[DType.bool], layout]:
        return self.elementwise_comparison[dt, Scalar[dt].__ge__](rhs)

    @always_inline
    fn __eq__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        rhs: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[DType.bool], layout]:
        return self.elementwise_comparison[dt, Scalar[dt].__eq__](rhs)

    @always_inline
    fn __ne__[
        dt: DType
    ](
        self: Tile[shape, Scalar[dt], layout],
        rhs: Tile[shape, Scalar[dt], layout],
    ) -> Tile[shape, Scalar[DType.bool], layout]:
        return self.elementwise_comparison[dt, Scalar[dt].__ne__](rhs)

    @always_inline
    fn to[
        dt: DType, t: DType
    ](self: Tile[shape, Scalar[t], layout]) -> Tile[shape, Scalar[dt], layout]:
        res = Tile[shape, Scalar[dt], layout]()

        @parameter
        for i in range(Int(Self.nelts)):
            res.impl[i] = self.impl[i].cast[dt]()

        return res

    @always_inline
    fn __getitem__[idx: UInt32](self) -> T:
        return self.impl[idx]


@always_inline
fn for_each_element_1d[
    LayoutType: Layout, //,
    layout: LayoutType,
    shape: Shape,
    f: fn[i: UInt32] (j: UInt32) capturing -> None,
](tid: UInt32):
    warp_id = tid // UInt32(32)
    lane_id = tid % UInt32(32)

    alias lane_size: UInt32 = layout.get_size_per_thread()[0]
    alias warp_size: UInt32 = lane_size * layout.get_threads_per_warp()[0]
    alias tile_size: UInt32 = warp_size * layout.get_warps_per_cta()[0]

    @parameter
    for k in range(Int(_layout_nelts(layout, shape))):
        alias reg_offset = k % layout.get_size_per_thread()[0]
        alias tile_id = k // layout.get_size_per_thread()[0]
        alias tile_offset = tile_id * tile_size
        alias shape0 = shape[0]
        warp_offset = warp_id * warp_size
        lane_offset = lane_id * lane_size
        offset = (warp_offset + lane_offset + reg_offset) % shape0
        f[k](tile_offset + offset)


@always_inline
fn arange[
    LayoutType: Layout, //, layout: LayoutType, lb: UInt32, ub: UInt32
]() -> Tile[(ub - lb), UInt32, layout]:
    res = Tile[(ub - lb), UInt32, layout]()

    @always_inline
    @parameter
    fn each[i: UInt32](j: UInt32):
        res.impl[i] = j + lb

    for_each_element_1d[layout, (ub - lb), each](thread_idx.x)

    return res


@always_inline
fn store[
    LayoutType: Layout, //,
    shape: Shape,
    T: AnyTrivialRegType,
    addrspace: UInt,
    layout: LayoutType,
](ptrs: Tile[shape, Ptr[T, addrspace], layout], value: Tile[shape, T, layout]):
    @parameter
    for i in range(Int(ptrs.nelts)):
        ptrs[i].store(value[i])


@always_inline
fn store[
    LayoutType: Layout, //,
    shape: Shape,
    T: AnyTrivialRegType,
    addrspace: UInt,
    layout: LayoutType,
](
    ptrs: Tile[shape, Ptr[T, addrspace], layout],
    value: Tile[shape, T, layout],
    mask: Tile[shape, Scalar[DType.bool], layout],
):
    @parameter
    for i in range(Int(ptrs.nelts)):
        if mask[i]:
            ptrs[i].store(value[i])


@always_inline
fn load[
    LayoutType: Layout, //,
    shape: Shape,
    T: AnyTrivialRegType,
    addrspace: UInt,
    layout: LayoutType,
](ptrs: Tile[shape, Ptr[T, addrspace], layout]) -> Tile[shape, T, layout]:
    res = Tile[shape, T, layout]()

    @parameter
    for i in range(Int(ptrs.nelts)):
        res.impl[i] = ptrs[i].load()

    return res


@always_inline
fn load[
    LayoutType: Layout, //,
    shape: Shape,
    T: AnyTrivialRegType,
    addrspace: UInt,
    layout: LayoutType,
](
    ptrs: Tile[shape, Ptr[T, addrspace], layout],
    mask: Tile[shape, Scalar[DType.bool], layout],
    other: T,
) -> Tile[shape, T, layout]:
    res = Tile[shape, T, layout]()

    @parameter
    for i in range(Int(ptrs.nelts)):
        if mask[i]:
            res.impl[i] = ptrs[i].load()
        else:
            res.impl[i] = other

    return res


@always_inline
fn where[
    LayoutType: Layout, //, shape: Shape, dt: DType, layout: LayoutType
](
    mask: Tile[shape, Scalar[DType.bool], layout],
    true_val: Tile[shape, Scalar[dt], layout],
    false_val: Tile[shape, Scalar[dt], layout],
) -> Tile[shape, Scalar[dt], layout]:
    res = Tile[shape, Scalar[dt], layout]()

    @parameter
    for i in range(Int(mask.nelts)):
        res.impl[i] = mask[i].select(true_val.impl[i], false_val.impl[i])

    return res


@always_inline
fn full[
    LayoutType: Layout, //, layout: LayoutType, shape: Shape, dt: DType
](value: Scalar[dt]) -> Tile[shape, Scalar[dt], layout]:
    res = Tile[shape, Scalar[dt], layout]()

    @parameter
    for i in range(Int(res.nelts)):
        res.impl[i] = value

    return res
