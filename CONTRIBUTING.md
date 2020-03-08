# Contributing to arim

Thank you for your interest in arim!
There are many way you can contribute to arim, including:

- reporting issues to fix bugs,
- writing documentation,
- writing example scripts to help users to get started quickly,
- writing unit tests to ensure the code works as intended,
- adding new features.

## Developer installation

A developer installation requires extra packages for testing, the documentation and code quality.
The recommended method is to use an [editable installation](https://pip.pypa.io/en/stable/reference/pip_install/), as follows:

1) Duplicate code from git repository
2) In the root directory which contains ```setup.py``, type in a console:
```
pip install -e .[dev]
```

The ``[dev]`` flags installs recommended packages for developement, as defined in ``extras_require`` in the file ``setup.py``.


## Code quality guidelines

Adhering to a common code style helps making the code more readable.

Code is written only once, but read multiple times, so writing readable code is essential to save time in the long run.
Meaningful variable and function names, with no or little abbreviation, are in particular essential.

arim code follows the general guidelines defined in [Python PEP-8](https://www.python.org/dev/peps/pep-0008/), with the amendments defined in the following sections.

### Code formatting

Python code must be formatted using [black](https://black.readthedocs.io/en/stable/).
To format a Python file named ``myfile.py``, type in a terminal:
```
black myfile.py
```

### Docstring

Docstrings of functions, classes and modules use [numpy's docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).

## Documentation

arim's documentation is powered by [Sphinx](http://sphinx-doc.org/).

The documentation is generated from two sources:

1. the ``.rst`` files in ``docs/source``, formatted in ReStructuredText,
2. the docstrings in arim's Python code, via [autosummary](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html)

For academic references, please use the author-date style from the Chicago Manual of Style, for example:

    Holmes, Caroline, Bruce W. Drinkwater, and Paul D. Wilcox. 2005. ‘Post-Processing of the Full Matrix of Ultrasonic Transmit–receive Array Data for Non-Destructive Evaluation’. NDT & E International 38 (8): 701–11. doi:10.1016/j.ndteint.2005.04.002.

### Building the documentation

1. Delete the cache of autosummary by deleting the directory ``docs/source/_autosummary``
2. In the directory ``docs``, type in a terminal:
```
make html
```

The output is in ``docs/source/html``.

## Version control discipline

A commit must contain one functional change. In other words a commit must not contain changes in several unrelated features.
[Always use informative commit messages](https://wiki.openstack.org/wiki/GitCommitMessages).

Please push in branch ``master`` only versions of arim which successfully pass all tests. When developing new complex features, please a create a new branch first, develop the feature, add tests, and finally create a pull request. to ``master`` when it is ready ([feature branch workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)).

## Pull request

You can propose changes to arim using *pull requests* ([Github's guide for pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests)).
By submitting a pull request, you accept that the proposed changes are licensed under the MIT license.
The proposed changes must comply with arim's code convention, as per previous section.

## Testing

Unit tests ensure that a given function returns intended results at the time of commit and later on (non-regression).
Unit testing in arim is powered by [pytest](https://docs.pytest.org).
The tests are defined in directory ``tests``. Consider adding new tests!

To run the tests, type in a terminal in the root directory which contains ``tests`` and ``arim``:

    pytest

All tests must pass.

## Releasing

To create a release:

1. Ensure all unit tests pass (see 'Testing' section)
2. Change arim's version number ``arim/__init__.py``, following [PEP 440](https://www.python.org/dev/peps/pep-0440/) convention, then commit with an instructive description such as "Release version 1.0"
3. Assign a [tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) to the release commit. The tag name should be "v1.0" if the version number is 0.1.
4. Build the documentation, save the HTML files in a zip file named "documentation.zip"
5. Create a wheel package with 

```
python setup.py bdist_wheel
```

The result is a `.whl` file in the directory ``dist``, for example ``arim-0.8-py3-none-any.whl``.

6. [Create a release on Github](https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository). Select the newly created tag. Describe the changes of this new version. Attach the ``documentation.zip`` file and the wheel (`.whl`) file to it.
