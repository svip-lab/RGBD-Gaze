class edict(dict):
    def __init__(self, d=None, **kwargs):
        super(edict, self).__init__()
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        # if isinstance(value, (list, tuple)):
        #     value = [self.__class__(x)
        #              if isinstance(x, dict) else x for x in value]
        # elif isinstance(value, dict) and not isinstance(value, self.__class__):
        #     value = self.__class__(value)
        super(edict, self).__setattr__(name, value)
        super(edict, self).__setitem__(name, value)

    __setitem__ = __setattr__
