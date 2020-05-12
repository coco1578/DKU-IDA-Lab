import sys
import logging
import argparse
import traceback
import utils as ut
import numpy as np
import common as cm

from log import wl_log
from crawler import Crawler


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Crawl a list of URLs in several batches.')

    parser.add_argument('-u', '--url_list', help='URL list file path')
    parser.add_argument('-b', '--browser_version', help='Tor browser version used to crawl, possible values are: 7.5.4', default='7.5.4')
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    parser.add_argument('--batch', help='Number of batches (default: %s)' % cm.NUM_BATCHES, default=cm.NUM_BATCHES)
    parser.add_argument('--instance', helgp='Number of instances (default: %s)' % cm.NUM_INSTANCES, default=cm.NUM_INSTANCES)
    parser.add_argument('--start', help='Crawl URLs after this line (1)')
    parser.add_argument('--stop', help='Crawl URLs until this line')
    parser.add_argument('-x', '--xvfb', help='Use XVFB (for headless testing)', action='store_true', default=False)
    parser.add_argument('-c', '--capture_screen', help='Capture page screenshots', action='store_true', default=False)

    args = parser.parse_args()

    url_list_path = args.url_list
    verbose = args.verbose
    tbb_version = args.browser_version
    no_of_batches = int(args.batch)
    no_of_instances = int(args.instance)
    start_line = int(args.start) if args.start else 1
    stop_line = int(args.stop) if args.stop else 999999999999
    xvfb = args.xvfb
    capture_screen = args.capture_screen

    if verbose:
        wl_log.setLevel(logging.DEBUG)
    else:
        wl_log.setLevel(logging.INFO)

    # Validate the given arguments
    # Read urls
    url_list = np.loadtxt(url_list_path, delimiter='\n', dtype=str)
    url_list = url_list.tolist()
    url_list = url_list[start_line-1:stop_line]
    torrc_dict = cm.TORRC_DEFAULT

    if not tbb_version:
        tbb_version = cm.TBB_DEFAULT_VERSION
    elif tbb_version not in cm.TBB_KNOWN_VERSIONS:
        ut.die('Version of Tor browser is not recognized.')

    crawler = Crawler(torrc_dict, url_list, tbb_version, xvfb, capture_screen)
    wl_log.info('Command line parameters: %s' % sys.argv)

    # Run the crawl
    try:
        crawler.crawl(no_of_batches, no_of_instances, start_line=start_line - 1)
    except KeyboardInterrupt:
        wl_log.warning('Keyboard interrupt! Quitting...')
    except Exception as e:
        wl_log.error('Exception: \n%s' % (traceback.format_exc()))
    finally:
        crawler.stop_crawl()