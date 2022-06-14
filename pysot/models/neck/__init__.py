# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.neck.neck import GoogLeNetAdjustLayer

NECKS = {
        'GoogLeNetAdjustLayer': GoogLeNetAdjustLayer,
        }

def get_neck(name, **kwargs):
    return NECKS[name](**kwargs)