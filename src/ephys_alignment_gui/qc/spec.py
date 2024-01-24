import re

from uuid import UUID
from typing import Union

SESSION_SPEC = '({lab}/Subjects/)?{subject}/{date}/{number}'
"""str: The session specification pattern"""

COLLECTION_SPEC = r'({collection}/)?(#{revision}#/)?'
"""str: The collection and revision specification pattern"""

FILE_SPEC = r'_?{namespace}?_?{object}\.{attribute}(?:_{timescale})?(?:\.{extra})*\.{extension}$'
"""str: The filename specification pattern"""

REL_PATH_SPEC = f'{COLLECTION_SPEC}{FILE_SPEC}'
"""str: The collection, revision and filename specification pattern"""

FULL_SPEC = f'{SESSION_SPEC}/{REL_PATH_SPEC}'
"""str: The full ALF path specification pattern"""

_DEFAULT = (
    ('lab', r'\w+'),
    ('subject', r'[\w.-]+'),
    ('date', r'\d{4}-\d{2}-\d{2}'),
    ('number', r'\d{1,3}'),
    ('collection', r'[\w./-]+'),
    ('revision', r'[\w.-]+'),  # brackets
    # to include underscores: r'(?P<namespace>(?:^_)\w+(?:_))?'
    ('namespace', '(?<=_)[a-zA-Z0-9]+'),  # brackets
    ('object', r'\w+'),
    # to treat _times and _intervals as timescale: (?P<attribute>[a-zA-Z]+)_?
    # (?:_[a-z]+_)? allows attribute level namespaces (deprecated)
    ('attribute', r'(?:_[a-z]+_)?[a-zA-Z0-9]+(?:_times(?=[_.])|_intervals(?=[_.]))?'),  # brackets
    ('timescale', r'\w+'),  # brackets
    ('extra', r'[.\w-]+'),  # brackets
    ('extension', r'\w+')
)

def _named(pattern, name):
    """Wraps a regex pattern in a named capture group"""
    return f'(?P<{name}>{pattern})'


def regex(spec: str = FULL_SPEC, **kwargs) -> re.Pattern:
    """
    Construct a regular expression pattern for parsing or validating an ALF

    Parameters
    ----------
    spec : str
        The spec string to construct the regular expression from
    kwargs : dict[str]
        Optional patterns to replace the defaults

    Returns
    -------
    re.Pattern
        A regular expression Pattern object

    Examples
    --------
    Regex for a filename

    >>> pattern = regex(spec=FILE_SPEC)

    Regex for a complete path (including root)

    >>> pattern = '.*' + regex(spec=FULL_SPEC).pattern

    Regex pattern for specific object name

    >>> pattern = regex(object='trials')
    """
    fields = dict(_DEFAULT)
    if not fields.keys() >= kwargs.keys():
        unknown = next(k for k in kwargs.keys() if k not in fields.keys())
        raise KeyError(f'Unknown field "{unknown}"')
    fields.update({k: v for k, v in kwargs.items() if v is not None})
    spec_str = spec.format(**{k: _named(fields[k], k) for k in re.findall(r'(?<={)\w+', spec)})
    return re.compile(spec_str)


def is_valid(filename):
    """
    Returns a True for a given file name if it is an ALF file, otherwise returns False

    Parameters
    ----------
    filename : str
        The name of the file to evaluate

    Returns
    -------
    bool
        True if filename is valid ALF

    Examples
    --------
    >>> is_valid('trials.feedbackType.npy')
    True
    >>> is_valid('_ns_obj.attr1.2622b17c-9408-4910-99cb-abf16d9225b9.metadata.json')
    True
    >>> is_valid('spike_train.npy')
    False
    >>> is_valid('channels._phy_ids.csv')  # WARNING: attribute level namespaces are deprecated
    True
    """
    return regex(FILE_SPEC).match(filename) is not None


def is_session_path(path_object):
    """
    Checks if the syntax corresponds to a session path.  Note that there is no physical check
    about existence nor contents

    Parameters
    ----------
    path_object : str, pathlib.Path
        The path object to validate

    Returns
    -------
    bool
        True if session path a valid ALF session path
    """
    session_spec = re.compile(regex(SESSION_SPEC).pattern + '$')
    if hasattr(path_object, 'as_posix'):
        path_object = path_object.as_posix()
    path_object = path_object.strip('/')
    return session_spec.search(path_object) is not None


def is_uuid_string(string: str) -> bool:
    """
    Bool test for randomly generated hexadecimal uuid validity.
    NB: unlike is_uuid, is_uuid_string checks that uuid is correctly hyphen separated
    """
    return isinstance(string, str) and is_uuid(string, (3, 4, 5)) and str(UUID(string)) == string


def is_uuid(uuid: Union[str, int, bytes, UUID], versions=(4,)) -> bool:
    """Bool test for randomly generated hexadecimal uuid validity.
    Unlike `is_uuid_string`, this function accepts UUID objects
    """
    if not isinstance(uuid, (UUID, str, bytes, int)):
        return False
    elif not isinstance(uuid, UUID):
        try:
            uuid = UUID(uuid) if isinstance(uuid, str) else UUID(**{type(uuid).__name__: uuid})
        except ValueError:
            return False
    return isinstance(uuid, UUID) and uuid.version in versions