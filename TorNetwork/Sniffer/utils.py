import re
import os
import sys
import signal
import psutil
import distutils.dir_util as du

from log import wl_log
from time import strftime
from urllib.request import urlopen
from hashlib import md5, sha256


class TimeExceededError(Exception):
    pass


def get_hash_of_directory(path):
    """
    Return md5 hash of the directory pointed by path.
    """
    m = md5()
    for root, _, files in os.walk(path):
        for f in files:
            full_path = os.path.join(root, f)
            for line in open(full_path).readlines():
                m.update(line)
    return m.digest()


def clone_dir_with_timestap(orig_dir_path):
    """Copy a folder into the same directory and append a timestamp."""
    new_dir = create_dir(append_timestamp(orig_dir_path))
    try:
        du.copy_tree(orig_dir_path, new_dir)
    except Exception as e:
        wl_log.error("Error while cloning the dir with timestamp" + str(e))
    finally:
        return new_dir


def create_dir(dir_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def append_timestamp(_str=''):
    """Append a timestamp to a string and return it."""
    return _str + strftime('%y%m%d_%H%M%S')


def die(last_words='Unknown problem, quitting!'):
    wl_log.error(last_words)
    sys.exit(1)


def raise_signal(signum, frame):
    raise TimeExceededError


def timeout(duration):
    """Timeout after given duration."""
    signal.signal(signal.SIGALRM, raise_signal)  # linux only !!!
    signal.alarm(duration)  # alarm after X seconds


def cancel_timeout():
    """Cancel a running alarm."""
    signal.alarm(0)


def get_filename_from_url(url, prefix):
    """Return base filename for the url."""
    url = url.replace('https://', '')
    url = url.replace('http://', '')
    url = url.replace('www.', '')
    dashed = re.sub(r'[^A-Za-z0-9._]', '-', url)
    return '%s-%s' % (prefix, re.sub(r'-+', '-', dashed))


def gen_all_children_procs(parent_pid):
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):
        yield child


def kill_all_children(parent_pid):
    """Kill all child process of a given parent."""
    for child in gen_all_children_procs(parent_pid):
        child.kill()


def read_file(path, binary=False):
    """Read and return the file content."""
    options = 'rb' if binary else 'rU'
    with open(path, options) as f:
        return f.read()


def sha_256_sum_file(path, binary=True):
    """Return the SHA-256 sum of the file."""
    return sha256(read_file(path, binary=binary)).hexdigest()


def gen_read_lines(path):
    """Generator for reading the lines in a file."""
    with open(path, 'rU') as f:
        for line in f:
            yield line


def read_url(uri):
    """Fetch and return a URI content."""
    w = urlopen(uri)
    return w.read()


def write_to_file(file_path, data):
    """Write data to file and close."""
    with open(file_path, 'w') as ofile:
        ofile.write(data)