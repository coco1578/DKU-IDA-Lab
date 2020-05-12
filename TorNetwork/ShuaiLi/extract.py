# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2020.02.17"


"""
import os
import sys
import glob
import dpkt
import json
import tqdm
import socket
import numpy as np
import configparser
import multiprocessing

from Preprocess import *


def extract(times, sizes, config):

    features = list()

    if config['DEFAULT']['PACKET_NUMBER']:
        pkt_number = pktNum.PktNumFeature(times, sizes).getPktNumFeature()

    if config['DEFAULT']['PKT_TIME']:
        pkt_time = time.Time(times, sizes).getTimeFeature()

    # if config['DEFAULT']['UNIQUE_PACKET_LENGTH']:
    #     unique_packet_length = pktLen.PktLen(times, sizes, features).getPktLenFeature()

    if config['DEFAULT']['NGRAM_ENABLE']:
        buckets_2 = ngram.Ngram(sizes, 2).getNgram()
        # features.extend(buckets)
        buckets_3 = ngram.Ngram(sizes, 3).getNgram()
        # features.extend(buckets)
        buckets_4 = ngram.Ngram(sizes, 4).getNgram()
        # features.extend(buckets)
        buckets_5 = ngram.Ngram(sizes, 5).getNgram()
        # features.extend(buckets)
        buckets_6 = ngram.Ngram(sizes, 6).getNgram()
        # features.extend(buckets)

    if config['DEFAULT']['TRANS_POSITION']:
        trans_position = transPosition.TransPosition(times, sizes).getTransPosFeature()

    if config['DEFAULT']['INTERVAL_KNN']:
        interval_knn = interval.Interval(sizes, 'KNN').getIntervalFeature()

    if config['DEFAULT']['INTERVAL_ICICS']:
        interval_icics = interval.Interval(sizes, 'ICICS').getIntervalFeature()

    if config['DEFAULT']['INTERVAL_WPES11']:
        interval_wpes11 = interval.Interval(sizes, 'WPES11').getIntervalFeature()

    if config['DEFAULT']['PACKET_DISTRIBUTION']:
        packet_distribution = pktDistribution.PktDistributiin(times, sizes).getPktDistFeature()

    if config['DEFAULT']['BURSTS']:
        bursts = burst.Burst(sizes).getBurstFeature()

    if config['DEFAULT']['FIRST20']:
        first20 = headtail.HeadTail(times, sizes).getFirst20()

    if config['DEFAULT']['FIRST30_PKT_NUM']:
        first30_pkt_num = headtail.HeadTail(times, sizes).getFirst30PktNum()

    if config['DEFAULT']['LAST30_PKT_NUM']:
        last30_pkt_num = headtail.HeadTail(times, sizes).getLast30PktNum()

    if config['DEFAULT']['PKT_PER_SECOND']:
        pkt_per_second = pktSec.PktSec(times, sizes, config['DEFAULT']['howlong']).getPktSecFeature()

    if config['DEFAULT']['CUMUL']:
        cum = cumul.Cumul(sizes, config['DEFAULT']['featureCount']).getCumulFeatures()

    features += pkt_number
    features += pkt_time
    # features.extend(unique_packet_length)
    features += buckets_2
    features += buckets_3
    features += buckets_4
    features += buckets_5
    features += buckets_6
    features += trans_position
    features += interval_knn
    features += interval_icics
    features += interval_wpes11
    features += packet_distribution
    features += bursts
    features += first20
    features += first30_pkt_num
    features += last30_pkt_num
    features += pkt_per_second
    features += cum

    return features


def getAtr(pcap_path, src_ip):
    fd = open(pcap_path, 'rb')
    pcap = dpkt.pcap.Reader(fd)

    times = list()
    sizes = list()
    features = list()

    try:
        for ts, buf in pcap:
            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data
            if eth.type != dpkt.ethernet.ETH_TYPE_IP:
                continue
            if ip.p != dpkt.ip.IP_PROTO_TCP:
                continue
            if socket.inet_ntoa(ip.src) == '192.168.160.10':
                continue
            if socket.inet_ntoa(ip.dst) == '192.168.160.10':
                continue
            if socket.inet_ntoa(ip.src) in src_ip:
                times.append(ts)
                sizes.append(1 * len(buf))
            else:
                times.append(ts)
                sizes.append(-1 * len(buf))

    except AttributeError:
        pass
    except dpkt.dpkt.UnpackError:
        pass
    except dpkt.dpkt.NeedData:
        pass

    fd.close()
    return times, sizes


def Main():

    config = configparser.ConfigParser()
    config.read('config.ini')

    tor_local_ip = ['192.168.160.100', '192.168.160.101', '192.168.160.102', '192.168.160.103', '192.168.160.104', '192.168.160.105',
                    '192.168.160.106', '192.168.160.107', '192.168.160.108', '192.168.160.109', '192.168.160.110']

    json_dict = json.load(open('/Volumes/SSD/TorNetwork/TBB/removeOutlierTraffic_TBB.json', 'r'))
    fd = open('/Users/coco/Desktop/TBB_07_10_ShuaiLi_IDA_new.spa', 'w')

    class_label = dict()
    for i, (webstie, pcap_pcap_list) in tqdm.tqdm(enumerate(json_dict.items())):
        if webstie not in class_label:
            class_label[webstie] = i
        for pcap_path in pcap_pcap_list:
            try:
                times, sizes = getAtr(pcap_path, tor_local_ip)
                features = extract(times, sizes, config)
                fd.write('%d ' % i)
                temp = list()
                for dim in range(len(features)):
                    temp.append('%d:%f' % (dim + 1, features[dim]))
                fd.write(' '.join(temp))
                fd.write('\n')
            except:
                continue
    fd.close()


if __name__ == '__main__':

    Main()
