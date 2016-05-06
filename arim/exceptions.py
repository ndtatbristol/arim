class ArimWarning(UserWarning):
    pass

class InvalidDimension(ValueError):
    """
    Raised when an array has an invalid dimension.
    """
    @classmethod
    def message_auto(cls, array_name, expected_dimension, current_dimension=None):
        current = ' (current: {})'.format(current_dimension) if current_dimension is not None else ''
        message = "Dimension of array '{}' must be {}{}.".format(array_name, expected_dimension, current)

        return cls(message)

class InvalidShape(ValueError):
    """
    Raised when an array has an invalid shape.
    """

    @classmethod
    def message_auto(cls, array_name, expected_shape, current_shape=None):
        current = ' (current: {})'.format(current_shape) if current_shape is not None else ''
        message = "Array '{}' must have shape {} (current: {}){}."\
            .format(array_name, expected_shape, current_shape, current)

        return cls(message)


class NotAnArray(TypeError):
    def __init__(self, array_name, message=None):
        if message is None:
            message = " '{}' must be an array. Try to convert to numpy.array first.".format(array_name)
        super().__init__(message)
