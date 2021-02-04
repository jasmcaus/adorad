import pytensor as ts
import pytest
import numpy as np


def test_tensor2f_init():
    tensor = ts.Tensor2F(2, 2)
    assert tensor.shape() == [2, 2]
    assert tensor.data_size() == 4


def test_tensor2f_getitem():
    tensor = ts.Tensor2F(2, 2)
    assert tensor[0, 0] == 0
    assert tensor[0, 1] == 0
    assert tensor[1, 0] == 0
    assert tensor[1, 1] == 0


def test_tensor2f_setitem():
    tensor = ts.Tensor2F(2, 2)
    assert tensor[1, 1] == 0
    tensor[1, 1] = 1337
    assert tensor[1, 1] == 1337


def test_tensor_from_numpy():
    with pytest.raises(RuntimeError) as e:
        ts.Tensor2F(np.array([1, 2, 3]))  # trying to assign a 1D array
    assert str(e.value) == "Incompatible buffer format!"

    array = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    tensor = ts.Tensor2F(array)

    for i in range(tensor.shape()[0]):
        for j in range(tensor.shape()[1]):
            assert tensor[i, j] == array[i, j]


def test_numpy_from_tensor():
    tensor = ts.Tensor2F(5, 4)
    assert memoryview(tensor).shape == (5, 4)

    assert tensor[2, 3] == 0

    tensor[2, 3] = 11.0
    tensor[3, 2] = 7.0
    assert tensor[2, 3] == 11
    assert tensor[3, 2] == 7

    array = np.array(tensor)
    assert array.shape == (5, 4)
    assert array[2, 3] == 11
    assert array[3, 2] == 7
    assert abs(array).sum() == 11 + 7

    array[2, 3] = 5
    assert array[2, 3] == 5
