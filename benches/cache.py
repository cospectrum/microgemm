Bytes = int
KiB = int
MiB = int

SIZE_OF_ELEMENT: Bytes = 4
KC = 1024

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

    print(f'const MC: usize = {mc};')
    print(f'const KC: usize = {kc=};')
    print(f'const NC: usize = {nc=};')


class Cache:
    _bytes: Bytes
    
    def __init__(self, bytes: Bytes) -> None:
        self._bytes = bytes
    
    def get_other_dim(self, dim: int, size_of_element: Bytes) -> int:
        size = self.num_of_elements(size_of_element)
        assert size % dim == 0
        return size // dim

    def num_of_elements(self, size_of_element: Bytes) -> int:
        assert self.bytes() % size_of_element == 0
        return self.bytes() // size_of_element

    def bytes(self) -> Bytes:
        return self._bytes

    def kib(self) -> KiB:
        bs = self.bytes()
        assert bs % 1024 == 0
        return bs * 1024

    def mib(self) -> int:
        kib = self.kib()
        assert kib % 1024 == 0
        return kib * 1024

    @classmethod
    def from_bytes(cls, b: Bytes) -> 'Cache':
        return cls(b)

    @classmethod
    def from_kibs(cls, k: KiB) -> 'Cache':
        return cls.from_bytes(k * 1024)

    @classmethod
    def from_mibs(cls, mibs: MiB) -> 'Cache':
        return cls.from_kibs(mibs * 1024)


if __name__ == "__main__":
    main()
