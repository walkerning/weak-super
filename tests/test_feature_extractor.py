# -*- coding: utf-8 -*-

import pytest

from wksuper.config import cfg_from_str
from wksuper.feature import FeatureExtractor

hog_test_cfgs = [
("""
[feature]
type = "hog"
win_size = 128
orientations = 9
pixels_per_cell = 8
cells_per_block = 3
""",
 {"dim": 15876}
),
("""
[feature]
type = "hog"
win_size = 64
orientations = 9
pixels_per_cell = 16
cells_per_block = 3
""",
 {"dim": 729})]

@pytest.fixture(params=hog_test_cfgs)
def hog_obj_and_results(request):
    cfg = cfg_from_str(request.param[0])
    print cfg
    return (FeatureExtractor.get_registry("hog")(cfg), request.param[1])
 
class TestHogFeatureExtractor(object):
    def test__calculate_hog_dim(self, hog_obj_and_results):
        hog_extractor = hog_obj_and_results[0]
        assert hog_extractor._calculate_hog_dim() == hog_obj_and_results[1]["dim"]
