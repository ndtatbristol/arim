import os

def get_data_filename(filename):
    """
    Returns absolute path of the data file 'filename'.

    Parameters
    ==========
    filename : str
        Filename relative to directory 'tests/data'


    """
    basedir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    return os.path.join(basedir, filename)



