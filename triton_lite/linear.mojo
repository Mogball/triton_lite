from buffer.dimlist import DimList
from collections import Dict, List
from utils.static_tuple import StaticTuple


trait Layout:
    fn get_elements_per_thread(self) -> Int:
        ...


@value
struct LinearLayout:
    alias BasesT = Dict[String, List[List[Int]]]

    var bases: Self.BasesT
    var outDims: Dict[String, Int]
    var surjective: Bool

    @staticmethod
    def identity1D(
        size: Int, inDim: String, outDim: String
    ) -> LinearLayout:
        if size == 0:
            return LinearLayout.empty()

        powersOf2 = List[List[Int]]()
        i = 1
        while i < size:
            e = List[Int]()
            e.append(i)
            powersOf2.append(e)
            i *= 2

        bases = Self.BasesT()
        bases[inDim] = powersOf2
        outDims = Dict[String, Int]()
        outDims[outDim] = size
        return LinearLayout(bases, outDims, True)

    fn getNumOutDims(self) -> Int:
        return len(self.outDims)

    def getOutDimNames(self) -> List[String]:
        ret = List[String]()
        for k in self.outDims.keys():
            ret.append(k[])
        return ret

    def getInDimNames(self) -> List[String]:
        ret = List[String]()
        for k in self.bases.keys():
            ret.append(k[])
        return ret

    def getOutDimSizeLog2(self, dimName: String) -> Int:
        return self.outDims[dimName]

    def getOutDimSize(self, dimName: String) -> Int:
        return 1 << self.getOutDimSizeLog2(dimName)

    def getOutDimIndex(self, dimName: String) -> Int:
        i = 0
        for name in self.outDims.keys():
            if name[] == dimName:
                return i
            i += 1

    def transposeOuts(self, newOutDims: List[String]) -> LinearLayout:
        permutation = List[Int]()
        for outDim in newOutDims:
            permutation.append(self.getOutDimIndex(outDim[]))

        newBases = Self.BasesT()
        for kv in self.bases.items():
            inDim = kv[].key
            inDimBases = kv[].value
            newInDimBases = List[List[Int]]()
            for basis in inDimBases:
                newBasis = List[Int]()
                for i in permutation:
                    newBasis.append(basis[][i[]])
                newInDimBases.append(newBasis)
            newBases[inDim] = newInDimBases

        newOutDimMap = Dict[String, Int]()
        for outDim in newOutDims:
            newOutDimMap[outDim[]] = self.getOutDimSize(outDim[])

        return LinearLayout(newBases, newOutDimMap, self.surjective)


fn findIndex[
    T: Copyable & Movable
](list: List[T], check: fn (T) -> Bool) -> Int:
    for i in range(len(list)):
        if check(list[i]):
            return i
    return -1


fn basesPerDimImpl(
    namedBases: LinearLayout.BasesT,
    dimName: String,
    rank: Int,
    skipBroadcast: Bool = True,
) raises -> List[Int]:
    bases = namedBases[dimName]

    ret = List[Int](rank, 1)

    if len(bases) == 0:
        return ret

    fn nonZero(val: Int) -> Bool:
        return val != 0

    nonZeroIdx = 0
    for basis in bases:
        it = findIndex(basis[], nonZero)
        if it != -1:
            nonZeroIdx = it
            ret[nonZeroIdx] *= 2
        elif not skipBroadcast:
            ret[nonZeroIdx] *= 2

    return ret


def ensureLayoutNotSmallerThan(
    layout: LinearLayout, shape: Dict[String, Int]
) -> LinearLayout:
    if len(shape) == 0:
        return layout

    kDim = layout.getInDimNames()[0]

    ret = layout
    for outDimName in layout.getOutDimNames():
        actualSize = layout.getOutDimSize(outDimName[])
        desiredSize = shape[outDimName[]]
        ret *= LinearLayout.identity1D(
            desiredSize // actualSize, kDim, outDimName[]
        )
    return ret


@value
struct LinearEncoding:
    var layout: LinearLayout

    fn getElemsPerThread(self) -> Int:
        pass

    def getOrder(self) -> List[Int]:
        rank = self.layout.getNumOutDims()
        order = List[Int](rank, 0)
        for i in range(rank):
            order[i] = rank - i - 1
        return order

    def getRepOrder(self) -> List[Int]:
        return self.getOrder()

    def toLinearLayout(self, shape: List[Int]) -> LinearLayout:
        ll = self.layout
        canonicalDims = ll.getOutDimNames()
        namedShape = Dict[String, Int]()
        permutedDims = List[String]()
        for dim in self.getRepOrder():
            permutedDims.append(canonicalDims[dim[]])
            namedShape[canonicalDims[dim[]]] = shape[dim[]]
        ll = ll.transposeOuts(permutedDims)
        ll = ensureLayoutNotSmallerThan(ll, namedShape)
        ll = ensureLayoutNotLargerThan(ll, namedShape, broadcastRegisters=False)
        ll = ll.transposeOuts(canonicalDims)
        return ll

    fn basesPerDim(
        self, dimName: String, skipBroadcast: Bool
    ) raises -> List[Int]:
        ll = self.layout
        rank = ll.getNumOutDims()
        return basesPerDimImpl(ll.bases, dimName, rank, skipBroadcast)


@register_passable("trivial")
struct Tile[shape: DimList, dtype: DType, layout: LinearLayout]:
    pass
