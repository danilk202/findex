import dataclasses
import pytest
from findex.models import Posting

def test_posting_is_frozen():
    p = Posting(1, 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.tf = 5

def test_posting_has_no_dict():
    assert not hasattr(Posting(1, 3), "__dict__")