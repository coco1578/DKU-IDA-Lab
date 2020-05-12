import re
import os
import sys
import glob
import json
import numpy as np

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


class HTMLParser:

    """
    15 features in html document
    ---------------------------------------
    | nol       : number of links         |
    | sdc       : same domain count       |
    | tpc       : third party count       |
    | notp      : number of tag path      |
    | nots      : number of tags          |
    | noup      : number of unique path   |
    | mind      : min depth               |
    | maxd      : max depth               |
    | ds        : depth sum               |
    | nomaxd    : number of max depth     |
    | nomind    : number of min depth     |
    | momediand : number of median depth  |
    | noc       : number of characters    |
    | nof       : number of files         |
    | hs        : html size               |
    ---------------------------------------
    """

    def __init__(self, file_path):

        self.file_path = file_path
        self.html_source, self.html_url = self.read_html_from_json()
        self.soup = self.set_beautifulsoup()

        self.title = self.html_title()
        self.depth_list = self.set_depth_list()
        self.tag_path = self.set_tag_path()
        self.tag = self.set_tag()

        self.nol = self.number_of_links()
        self.sdc = self.same_domain_count()
        self.tpc = self.third_party_count()
        self.notp, self.nots, self.noup, self.mind, self.maxd, self.ds, self.nomaxd, self.nomind, self.nomediand = self.get_depth_attr()
        self.noc = self.number_of_characters()
        self.nof = self.number_of_files()
        self.hs = self.html_size()

    def read_html_from_json(self):
        """
        json_file type:
        {
            'html': '<!doctype html><html></html>',
            'url': 'https://www.google.com'
        }
        """
        try:
            fd = open(self.file_path)
            json_file = json.load(fd)
        except FileExistsError as ferr:
            print('File does not exists!!', ferr)
            sys.exit(1)

        return json_file['html'], json_file['url']

    def set_beautifulsoup(self):
        soup = BeautifulSoup(self.html_source, 'html.parser')

        return soup

    def _get_uri(self, href):
        url = self.html_url

        if href.startswith('#'):
            return None

        absoulte_url = urljoin(url, href)

        return absoulte_url

    def number_of_links(self):
        try:
            domain_list = list()

            for link in self.soup.find_all('a', href=True):
                _uri = link.get('href')

                uri = self._get_uri(_uri)
                if uri is not None:
                    domain_list.append(uri)

            return str(len(domain_list))

        except ValueError as ve:
            print('There is no links in html file', ve)
            return 0

    def same_domain_count(self):
        try:
            domain_list = list()

            for link in self.soup.find_all('a', href=True):
                _uri = link.get('href')

                uri = self._get_uri(_uri)
                if uri is not None:
                    domain_list.append(uri)

            unique_domain_list, count = np.unique(
                domain_list, return_counts=True)
            print('Unique domain list', unique_domain_list)
            print('Length of unique domain list', count)

            return str(np.sum(count[np.where(count > 1)]))

        except ValueError as ve:
            print('There is no same domain links in html file.', ve)
            return 0

    def set_depth_list(self):
        depth_list = list()

        for element in self.soup.findAll():
            path = str(
                '.'.join(reversed([tag.name for tag in element.parentGenerator() if tag])))
            path += '.' + str(element.name)
            path = path.split('.')[1:]

            depth = len(path)
            depth_list.append(depth)

        depth_list = np.array(depth_list)

        return depth_list

    def set_tag_path(self):
        tag_path = list()

        try:
            for element in self.soup.findAll():
                path = str(
                    '.'.join(reversed([tag.name for tag in element.parentGenerator() if tag])))
                path += '.' + str(element.name)
                path = path.split('.')[1:]

                tag_path.append(path)

            tag_path = np.array(tag_path)

            return tag_path

        except ValueError as ve:
            print('There is no tag path in html file', ve)
            return 0

    def set_tag(self):
        tag = list()

        try:
            for element in self.soup.findAll():
                tag.append(element.name)

            tag = np.array(tag)

            return tag

        except ValueError as ve:
            print('There is no tag in html file', ve)
            return 0

    def get_depth_attr(self):
        tag = self.tag
        tag_path = self.tag_path
        depth_list = self.depth_list

        number_of_tags_path, number_of_tags, number_of_unique_path = len(
            tag_path), len(tag), len(np.unique(tag_path))

        min_depth, max_depth, depth_sum, median_depth = min(depth_list), max(
            depth_list), sum(depth_list), int(np.median(depth_list))

        number_of_min_depth, number_of_max_depth, number_of_median_depth = len(depth_list[np.where(depth_list == min_depth)]), len(
            depth_list[np.where(depth_list == max_depth)]), len(depth_list[np.where(depth_list == median_depth)])

        return str(number_of_tags_path), str(number_of_tags), str(number_of_unique_path), str(min_depth), str(max_depth), str(depth_sum), str(number_of_max_depth), str(number_of_min_depth), str(number_of_median_depth)

    def number_of_characters(self):
        if self.soup.body:
            body = self.soup.body

            if body.select('script'):
                for tag in body.select('script'):
                    tag.decompose()

            if body.select('style'):
                for tag in body.select('style'):
                    tag.decompose()

            text = body.get_text()
            text = text.rstrip().rsplit()

            txt_length = [len(txt) for txt in text]

            return str(sum(txt_length))

        else:
            print('There is no body tag in html file')
            pattern = '>(.+?)<'

            temp_list = re.findall(pattern, str(self.soup.get_text))
            length = list()

            for item in temp_list:
                item = str(item)
                item = item.rstrip().rsplit()
                length.append(len(item))

            length = np.array(length)

            return np.sum(length)

    def number_of_files(self):
        try:
            pattern = r'(src=\".+?[\.jpg|\.png|\.gif|\.bmp]\")'
            file_list = re.findall(pattern, str(self.soup.get_text))

            return str(len(file_list))

        except ValueError as ve:
            print('There is no files in html file', ve)
            return 0

    def html_size(self):
        html_size = os.path.getsize(self.file_path)

        return str(html_size)

    def third_party_count(self):
        thrid_party_list = list()

        base_url_host = urlparse(self.html_url).hostname

        try:
            for link in self.soup.find_all('a', href=True):
                uri = link.get('href')
                if uri.startswith('#'):
                    continue

                uri = self._get_uri(uri)
                if uri is not None:
                    domain_host = urlparse(uri).hostname

                    if base_url_host != domain_host:
                        thrid_party_list.append(uri)

            return str(len(thrid_party_list))

        except ValueError as ve:
            print('There is no thrid party domain', ve)
            return 0

    def html_title(self):

        try:
            pattern = '>(.+?)<'

            return re.findall(pattern, str(self.soup.title))[0]

        except ValueError as ve:
            print('There is no title in html file', ve)
            return "None"

    def __repr__(self):

        return 'number of links : {}\nsame domain count : {}\nthird party count : {}\nnumber of tag path : {}\
        \nnumber of tag : {}\nnumber of unique path : {}\nmin depth : {}\nmax depth : {}\ndepth sum : {}\
        \nnumber of max depth : {}\nnumber of min depth : {}\nnumber of median depth : {}\nnumber of characters : {}\
        \nnumber of files : {}\nhtml size : {}\n'.format(
            self.nol, self.sdc, self.tpc, self.notp, self.nots, self.noup,
            self.mind, self.maxd, self.ds, self.nomaxd, self.nomind,
            self.nomediand, self.noc, self.nof, self.hs)
