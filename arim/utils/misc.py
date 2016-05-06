__all__ = ['get_name']


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
