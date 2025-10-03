"""SingletonMeta"""


class SingletonMeta(type):
    "Creates a singleton, ensuring only one instance of the class is ever created"

    _instances = {}

    def __call__(cls, *args, **kwds):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwds)
            cls._instances[cls] = instance
        return cls._instances[cls]
