Bytes = int
KiB = int
MiB = int

SIZE_OF_ELEMENT: Bytes = 4

MR = 4
NR = 4


def main() -> None:
    # sysctl -a | grep 'cachesize'
    l1 = Cache.from_bytes(65536)
    l2 = Cache.from_bytes(4194304)
    l3 = Cache.from_mibs(8)

    kc = l1.get_other_dim(NR, SIZE_OF_ELEMENT)
    mc = l2.get_other_dim(kc, SIZE_OF_ELEMENT)
    nc = l3.get_other_dim(kc, SIZE_OF_ELEMENT)

    print(f"const MC: usize = {mc};")
    print(f"const KC: usize = {kc};")
    print(f"const NC: usize = {nc};")


class Cache:
    _bytes: Bytes

    def __init__(self, bytes: Bytes) -> None:
        self._bytes = bytes

    def get_other_dim(self, dim: int, size_of_element: Bytes) -> int:
        size = self.num_of_elements(size_of_element)
        return div(size, dim)

    def num_of_elements(self, size_of_element: Bytes) -> int:
        return div(self.byte(), size_of_element)

    def byte(self) -> Bytes:
        return self._bytes

    def kib(self) -> KiB:
        return div(self.byte(), 1024)

    def mib(self) -> MiB:
        return div(self.kib(), 1024)

    @classmethod
    def from_bytes(cls, b: Bytes) -> "Cache":
        return cls(b)

    @classmethod
    def from_kibs(cls, kib: KiB) -> "Cache":
        return cls.from_bytes(kib * 1024)

    @classmethod
    def from_mibs(cls, mib: MiB) -> "Cache":
        return cls.from_kibs(mib * 1024)


def div(left: int, right: int) -> int:
    assert left % right == 0
    return left // right


if __name__ == "__main__":
    main()
