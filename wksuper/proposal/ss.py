# -*- coding: utf-8 -*-

from wksuper.proposal.proposaler import Proposaler

class SSProposaler(Proposaler):
    TYPE = "ss"

    def __init__(self, cfg, **kwargs):
        super(SSProposaler, self).__init__(cfg, **kwargs)

    def make_proposal(self, im):
        """
        TODO
        ------------
        # output rois [[x0, y0, height0, width0],[x, y, height, width], ...]
        # Attention! [x0, y0, height, width] is an image up to 4% border, must be putted in the first line
        """
        raise NotImplementedError()
