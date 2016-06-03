# -*- coding: utf-8 -*-

from .exceptions import (ClassDefinitionError,
                         RegistryLookupError)

class meta(type):
    def __init__(cls, name, bases, dct):
        if not hasattr(cls, '_registry'):
            cls._registry = {}
        if "TYPE" not in dct:
            raise ClassDefinitionError("'TYPE' field must be in the class")
        _type = dct["TYPE"]
        if _type is not None:
            cls._registry[_type] = cls
            print "{}: {} is registered as TYPE '{}'".format(cls._ROLE, name, _type)
            
        super(meta, cls).__init__(name, bases, dct)

    def get_registry(cls, name):
        if name not in cls._registry:
            raise RegistryLookupError("{}: TYPE '{}' not registered yet".format(cls._ROLE, name))
        else:
            print "get_registry: {}: Use TYPE '{}'".format(cls._ROLE, name)
            return cls._registry[name]
