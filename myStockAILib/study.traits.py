# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:59:17 2017

@author: docul
"""

import traits.api as trapi
import traitsui.api as trui

class short_rate(trapi.HasTraits):
    name = trapi.Str
    rate = trapi.Float
    check = trapi.Bool
    
c = short_rate()
c.configure_traits()
