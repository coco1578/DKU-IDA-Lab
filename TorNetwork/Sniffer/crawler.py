from log import wl_log, add_log_file_handler, add_symlink
from random import choice
from selenium.common.exceptions import TimeoutException
from torutils import TorController
from visit import Visit
import common as cm
import os
import time
import utils as ut


class Crawler(object):
    """ Provides methods to collect traffic traces. """

    def __init__(self, torrc_dit, url_list, tbb_version, xvfb=False, capture_screen=True):
        self.crawl_dir = None
        self.crawl_logs_dir = None
        self.visit = None
        self.urls = url_list
        self.init_crawl_dirs()
        self.tor_log = os.path.join(self.crawl_logs_dir, 'tor.log')
        linkname = os.path.join(cm.RESULTS_DIR, 'latest_tor_log')
        add_symlink(linkname, self.tor_log)
        self.tbb_version = tbb_version
        self.tor_controller = TorController(
            torrc_dit, tbb_version, self.tor_log)
        self.tor_process = None
        self.tb_driver = None
        self.capture_screen = capture_screen
        self.xvfb = xvfb
        add_log_file_handler(wl_log, self.log_file)
        linkname = os.path.join(cm.RESULTS_DIR, 'latest_crawl_log')
        add_symlink(linkname, self.log_file)

    def crawl(self, num_batches=cm.NUM_BATCHES, num_instances=cm.NUM_INSTANCES, start_line=0):
        wl_log.info('Crawl configuration: batches %s, instances: %s, tbb_version %s, no of URLs: %s, crawl dir: %s, XVFB: %s, screenshot: %s'
                    % (num_batches, num_instances, self.tbb_version, len(self.urls), self.crawl_dir, self.xvfb, self.capture_screen))

        # for each batch
        for batch_num in range(num_batches):
            wl_log.info('********** Starting batch %s **********' % batch_num)
            site_num = start_line
            bg_site = None
            batch_dir = ut.create_dir(
                os.path.join(self.crawl_dir, str(batch_num)))

            # init/reset tor process to have a different circuit.
            # make sure that we're not using the same guard node again
            wl_log.info('********** Restarting Tor Before Batch **********')
            self.tor_controller.restart_tor()
            sites_crawled_with_same_proc = 0

            # for each site
            for page_url in self.urls:
                sites_crawled_with_same_proc += 1
                if sites_crawled_with_same_proc > cm.MAX_SITES_PER_TOR_PROCESS:
                    wl_log.info('********** Restarting Tor Process **********')
                    self.tor_controller.restart_tor()
                    sites_crawled_with_same_proc = 0

                wl_log.info('********** Crawling %s **********' % page_url)
                page_url = page_url[:cm.MAX_FNAME_LENGTH]
                site_dir = ut.create_dir(os.path.join(
                    batch_dir, ut.get_filename_from_url(page_url, site_num)))

                for instance_num in range(num_instances):
                    wl_log.info('********** Visit #%s to %s **********' %
                                (instance_num, page_url))
                    self.visit = None
                    try:
                        self.visit = Visit(batch_num, site_num, instance_num, page_url, site_dir,
                                           self.tbb_version, self.tor_controller, bg_site, self.xvfb, self.capture_screen)
                        self.visit.get()
                    except KeyboardInterrupt:  # CTRL + C
                        raise KeyboardInterrupt
                    except (ut.TimeExceededError, TimeoutException) as exc:
                        wl_log.critical('Visit to %s timed out! %s %s' % (
                            page_url, exc, type(exc)))
                        if self.visit:
                            self.visit.cleanup_visit()
                    except Exception:
                        wl_log.critical('Exception crawling %s' %
                                        page_url, exc_info=True)
                        if self.visit:
                            self.visit.cleanup_visit()

                # END - for each visit
                site_num += 1
                time.sleep(cm.PAUSE_BETWEEN_SITES)

    def init_crawl_dirs(self):
        """ Create results and logs directions for this crawl. """
        self.crawl_dir, self.crawl_logs_dir = self.create_crawl_dir()
        sym_link = os.path.join(cm.RESULTS_DIR, 'latest')
        add_symlink(sym_link, self.crawl_dir)  # add a symbolic link
        # Create crawl log
        self.log_file = os.path.join(self.crawl_logs_dir, 'crawl.log')

    def init_logger(self):
        """ Configure logging for crawler. """
        add_log_file_handler(wl_log, self.log_file)

    def stop_crawl(self):
        """ Cleans up crawl and kills tor process in case it's running. """
        wl_log.info('Stopping crawl...')
        if self.visit:
            self.visit.cleanup_visit()
        self.tor_controller.kill_tor_proc()

    def create_crawl_dir(self):
        """ Create a timestamped crawl. """
        ut.create_dir(cm.RESULTS_DIR)
        crawl_dir_wo_ts = os.path.join(cm.RESULTS_DIR, 'crawl')
        crawl_dir = ut.create_dir(ut.append_timestamp(crawl_dir_wo_ts))
        crawl_logs_dir = os.path.join(crawl_dir, 'logs')
        ut.create_dir(crawl_logs_dir)
        return crawl_dir, crawl_logs_dir