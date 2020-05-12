import os
import platform


class TBBTarballVerificationError(Exception):
    pass


class TBBSigningKeyImportError(Exception):
    pass


class TBBGetRecommendedVersionError(Exception):
    pass


class DumpcapTimeoutError(Exception):
    pass


env_vars = os.environ

architecture = platform.architecture()
if '64' in architecture[0]:
    arch = '64'
    machine = 'x86_64'
elif '32' in architecture[0]:
    arch = '32'
    machine = 'i686'
else:
    raise RuntimeError('Architecture is not known: %s' % architecture)

# shortcuts
path = os.path
join = path.join
dirname = os.path.dirname
expanduser = os.path.expanduser

# timeouts and pauses
PAUSE_BETWEEN_SITES = 5      # pause before crawling a new site
WAIT_IN_SITE = 5             # time to wait after the page loads
PAUSE_BETWEEN_INSTANCES = 4  # pause before visiting the same site (instances)
SOFT_VISIT_TIMEOUT = 120     # timeout used by selenium and dumpcap
# signal based hard timeout in case soft timeout fails
HARD_VISIT_TIMEOUT = SOFT_VISIT_TIMEOUT + 10
# max dumpcap size in KB
MAX_DUMP_SIZE = 30000
# max filename length
MAX_FNAME_LENGTH = 200

DISABLE_RANDOMIZEDPIPELINENING = False  # use with caution!
STREAM_CLOSE_TIMEOUT = 20  # wait 20 seconds before raising an alarm signal

XVFB_W = 1280
XVFB_H = 720

# Tor browser version suffixes
TBB_V_7_5_4 = '7.5.4'
TBB_DEFAULT_VERSION = TBB_V_7_5_4
TBB_KNOWN_VERSIONS = [TBB_V_7_5_4]

# Default paths
BASE_DIR = path.abspath(os.path.dirname(__file__))
DATASET_DIR = join(BASE_DIR, "datasets")
RESULTS_DIR = join(BASE_DIR, 'results')
ETC_DIR = join(BASE_DIR, 'etc')
PERMISSIONS_DB = join(ETC_DIR, 'permissions.sqlite')
HOME_PATH = expanduser('~')
TBB_BASE_DIR = join(BASE_DIR, 'tbb')

# Tor ports
SOCKS_PORT = 9050
CONTROLLER_PORT = 9051
MAX_ENTRY_GUARDS = "1"

# defaults for batch and instance numbers
NUM_BATCHES = 10
NUM_INSTANCES = 4
MAX_SITES_PER_TOR_PROCESS = 100  # reset tor process after crawling 100 sites

# torrc dictionaries
TORRC_DEFAULT = {'socksport': str(SOCKS_PORT),
                 'controlport': str(CONTROLLER_PORT)}

TORRC_WANG_AND_GOLDBERG = {'socksport': str(SOCKS_PORT),
                           'controlport': str(CONTROLLER_PORT),
                           'MaxCircuitDirtiness': '600000',
                           'UseEntryGuards': '0'
                           }

# Directory structure and paths depend on TBB versions
# Path to Firefox binary in TBB dir
TBB_V7_FF_BIN_PATH = join('Browser', 'firefox')
TBB_FF_BIN_PATH_DICT = {'7': TBB_V7_FF_BIN_PATH}

# Path to Firefox profile in TBB dir
TBB_V7_PROFILE_PATH = join('Browser', 'TorBrowser', 'Data', 'Browser', 'profile.default')
TBB_PROFILE_DIR_DICT = {'7': TBB_V7_PROFILE_PATH}

# Path to Tor binary in TBB dir
TOR_V7_BINARY_PATH = join('Browser', 'TorBrowser', 'Tor', 'tor')
TOR_BINARY_PATH_DICT = {'7': TOR_V7_BINARY_PATH}

# Path to Tor data in TBB dir
TOR_V7_DATA_DIR = join('Browser', 'TorBrowser', 'Data', 'Tor')
TOR_DATA_DIR_DICT = {'7': TOR_V7_DATA_DIR}


def get_tbb_major_version(version):
    return version.split('.')[0]


def get_tbb_dirname(version, os_name='linux', lang='en-US'):
    return 'tor-browser-%s%s-%s_%s' % (os_name, arch, version, lang)


def get_tbb_path(version, os_name='linux', lang='en-US'):
    dirname = get_tbb_dirname(version, os_name, lang)
    return join(TBB_BASE_DIR, dirname)


def get_tb_bin_path(version, os_name='linux', lang='en-US'):
    major = get_tbb_major_version(version)
    bin_path = TBB_FF_BIN_PATH_DICT[major]
    dir_path = get_tbb_path(version, os_name, lang)
    return join(dir_path, bin_path)


def get_tor_bin_path(version, os_name='linux', lang='en-US'):
    major = get_tbb_major_version(version)
    bin_path = TOR_BINARY_PATH_DICT[major]
    dir_path = get_tbb_path(version, os_name, lang)
    return join(dir_path, bin_path)


def get_tbb_profile_path(version, os_name='linux', lang='en-US'):
    major = get_tbb_major_version(version)
    profile = TBB_PROFILE_DIR_DICT[major]
    dir_path = get_tbb_path(version, os_name, lang)
    return join(dir_path, profile)


def get_tor_data_path(version, os_name='linux', lang='en-US'):
    major = get_tbb_major_version(version)
    data_path = TOR_DATA_DIR_DICT[major]
    tbb_path = get_tbb_path(version, os_name, lang)
    return join(tbb_path, data_path)