from ds import *


def test_mnist():
    mnist = MNIST()
    print(mnist[2])
    print(len(mnist))


def test_cub200():
    cub = CUB200()
    print(cub[2])
    print(len(cub))


def test_nli():
    nli = AllNLI()
    print(nli[2])
    print(len(nli))


def test_code():
    code = Code()
    print(code[2])
    print(len(code))


def test_test_split_exists():
    cls = [MNIST, CUB200, AllNLI, Code, MNISTParity, SST2]
    for cl in cls:
        ds = cl("test")
        print(len(ds))
