from .layout import Layout

from sys import is_amd_gpu, is_nvidia_gpu


@value
struct Blocked(Layout):
    var sizePerThread: List[UInt]
    var threadsPerWarp: List[UInt]
    var warpsPerCTA: List[UInt]

    @staticmethod
    fn one_d(sizePerThread: UInt, num_warps: UInt) -> Blocked:
        sz = List[UInt]()
        sz.append(sizePerThread)
        tpw = List[UInt]()
        tpw.append(32)
        nws = List[UInt]()
        nws.append(num_warps)
        return Blocked(
            sizePerThread=sz,
            threadsPerWarp=tpw,
            warpsPerCTA=nws,
        )

    fn get_size_per_thread(self) -> List[UInt]:
        return self.sizePerThread

    fn get_threads_per_warp(self) -> List[UInt]:
        return self.threadsPerWarp

    fn get_warps_per_cta(self) -> List[UInt]:
        return self.warpsPerCTA

    fn get_total_size_per_thread(self, shape: Shape) -> List[UInt]:
        ret = List[UInt]()
        for i in range(shape.size()):
            nelts = shape[i]
            nregs = (
                self.warpsPerCTA[i]
                * self.threadsPerWarp[i]
                * self.sizePerThread[i]
            )
            if nelts > nregs:
                ret.append(nelts // nregs)
            else:
                ret.append(self.sizePerThread[i])
        return ret
