import subprocess
import os

def get_git_version(short=True):
    """
    Returns the current git revision as a string. Returns an empty string
    if git is not available or if the library is not not in a repository.
    """
    curdir = os.getcwd()
    filedir, _ = os.path.split(__file__)
    os.chdir(filedir)

    if short:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
    else:
        cmd = ['git', 'rev-parse', 'HEAD']

    try:
        githash = subprocess.check_output(cmd)
        githash = githash.decode('ascii').strip()
    except (FileNotFoundError, subprocess.CalledProcessError)  as e:
        githash = ''

    os.chdir(curdir)
    return githash
