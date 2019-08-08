import pytest

import src.util.mod as mod

@pytest.mark.parametrize(
    "int1,int2,expected",
    [(1,5,1), (2,5,3), (3,17,6), (2,21,11)]
)
def test_mod_inverse(int1,int2,expected):
    assert mod.mod_inverse(int1,int2) == expected