__all__ = ['get_name', 'parse_enum_constant']


def get_name(metadata):
    """Return the name of an object based on the dictionary metadata. By preference: long_name, short_name, 'Unnamed'
    """
    name = metadata.get('long_name', None)
    if name is not None:
        return name

    name = metadata.get('short_name', None)
    if name is not None:
        return name

    return 'Unnamed'


def parse_enum_constant(enum_constant_or_name, enum_type):
    """
    Return the enumerated constant corresponding to 'enum_constant_or_name', which
    can be either this constant or a its name (string).
    """
    if isinstance(enum_constant_or_name, enum_type):
        return enum_constant_or_name
    else:
        try:
            return enum_type[enum_constant_or_name]
        except KeyError:
            raise ValueError("Expected a constant of enum '{enum_type}', got '{x}' instead".format(x=enum_constant_or_name, enum_type=enum_type))
