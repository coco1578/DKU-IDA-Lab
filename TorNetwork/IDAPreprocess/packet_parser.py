# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2019-08-26"

__last modified by__ = "coco"
__last modified time__ = "2019-09-25"


"""
import os
import json
import dpkt
import tqdm
import socket
import itertools
import numpy as np

from scipy.stats import skew
from scapy.all import rdpcap


class flowGenerator:

    def __init__(self, pcap_path, ip):

        # initializer
        self._ip = ip
        self.total_cell, self.total_time_list, self.out_time_list, self.in_time_list = self.readPcap(pcap_path)

    def readPcap(self, pcap_path):

        tor_cell_list = list()
        out_time_list = list()
        in_time_list = list()
        total_time_list = list()

        pcap = dpkt.pcap.Reader(open(pcap_path, 'rb'))

        try:
            for timestamp, buf in pcap:
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

                # Start preprocessing
                if socket.inet_ntoa(ip.src) in self._ip:
                    if len(buf) % 600 == 0:
                        total_time_list.append(timestamp)
                        out_time_list.append(timestamp)
                        for _ in range(int(len(buf) // 600)):
                            tor_cell_list.append(1)
                    else:
                        total_time_list.append(timestamp)
                        out_time_list.append(timestamp)
                        for _ in range(int(len(buf) // 600) + 1):
                            tor_cell_list.append(1)
                else:
                    if len(buf) % 600 == 0:
                        total_time_list.append(timestamp)
                        in_time_list.append(timestamp)
                        for _ in range(int(len(buf) // 600)):
                            tor_cell_list.append(-1)
                    else:
                        total_time_list.append(timestamp)
                        in_time_list.append(timestamp)
                        for _ in range(int(len(buf) // 600) + 1):
                            tor_cell_list.append(-1)
        except AttributeError:
            pass
        except dpkt.dpkt.UnpackError:
            pass
        except dpkt.dpkt.NeedData:
            pass

        return tor_cell_list, total_time_list, out_time_list, in_time_list

    def getCellLength(self):

        # mean, std, var, skew
        return np.mean(self.total_cell), np.std(self.total_cell), np.var(self.total_cell), skew(self.total_cell)

    def getCellIAT(self):

        # min, max, mean, median, std, var, quantile(0.25, 0.75), skew
        iat_list = []
        out_iat_list = []
        in_iat_list = []

        total_time_list = list(set(self.total_time_list))
        out_time_list = list(set(self.out_time_list))
        in_time_list = list(set(self.in_time_list))

        for i, time in enumerate(total_time_list):
            if i == 0:
                continue
            _iat_time = total_time_list[i] - total_time_list[i - 1]
            iat_list.append(_iat_time)

        for i, time in enumerate(out_time_list):
            if i == 0:
                continue
            _iat_time = out_time_list[i] - out_time_list[i - 1]
            out_iat_list.append(_iat_time)

        for i, time in enumerate(in_time_list):
            if i == 0:
                continue
            _iat_time = in_time_list[i] - in_time_list[i - 1]
            in_iat_list.append(_iat_time)

        return np.min(iat_list), np.max(iat_list), np.mean(iat_list), np.median(iat_list), np.std(iat_list), np.var(
            iat_list), np.quantile(iat_list, 0.25), np.quantile(iat_list, 0.75), skew(iat_list), \
               np.min(out_iat_list), np.max(out_iat_list), np.mean(out_iat_list), np.median(out_iat_list), np.std(
            out_iat_list), np.var(out_iat_list), np.quantile(out_iat_list, 0.25), np.quantile(out_iat_list, 0.75), skew(
            out_iat_list), \
               np.min(in_iat_list), np.max(in_iat_list), np.mean(in_iat_list), np.median(in_iat_list), np.std(
            in_iat_list), np.var(in_iat_list), np.quantile(in_iat_list, 0.25), np.quantile(in_iat_list, 0.75), skew(
            in_iat_list)

    def getCellBurst(self):

        idx_list = []

        start = True  # if 1, if False -1

        for i in range(len(self.total_cell) - 1):
            if self.total_cell[0] == 1:
                start = True
            else:
                start = False

            if self.total_cell[i] != self.total_cell[i + 1]:
                idx_list.append(i + 1)

        if idx_list[-1] != len(self.total_cell):
            idx_list.append(len(self.total_cell))

        ori_burst = [idx_list[0]] + np.diff(idx_list).tolist()
        if start is True:
            out_burst = ori_burst[0::2]
            in_burst = ori_burst[1::2]
        else:
            out_burst = ori_burst[1::2]
            in_burst = ori_burst[0::2]

        return np.max(ori_burst), np.mean(ori_burst), np.median(ori_burst), np.std(ori_burst), np.var(
            ori_burst), np.quantile(ori_burst, 0.25), np.quantile(ori_burst, 0.75), skew(ori_burst), \
               np.max(out_burst), np.mean(out_burst), np.median(out_burst), np.std(out_burst), np.var(
            out_burst), np.quantile(out_burst, 0.25), np.quantile(out_burst, 0.75), skew(out_burst), \
               np.max(in_burst), np.mean(in_burst), np.median(in_burst), np.std(in_burst), np.var(
            in_burst), np.quantile(in_burst, 0.25), np.quantile(in_burst, 0.75), skew(in_burst),

    def get30CellBurst(self):

        idx_list = []

        start = True  # if 1, if False -1

        first30 = self.total_cell[:30]

        for i in range(len(first30) - 1):
            if first30[0] == 1:
                start = True
            else:
                start = False

            if first30[i] != first30[i + 1]:
                idx_list.append(i + 1)

        if idx_list[-1] != len(first30):
            idx_list.append(len(first30))

        ori_burst = [idx_list[0]] + np.diff(idx_list).tolist()
        if start is True:
            out_burst = ori_burst[0::2]
            in_burst = ori_burst[1::2]
        else:
            out_burst = ori_burst[1::2]
            in_burst = ori_burst[0::2]

        return np.max(ori_burst), np.mean(ori_burst), np.median(ori_burst), np.std(ori_burst), np.var(
            ori_burst), np.quantile(ori_burst, 0.25), np.quantile(ori_burst, 0.75), skew(ori_burst), \
               np.min(out_burst), np.max(out_burst), np.mean(out_burst), np.median(out_burst), np.std(
            out_burst), np.var(out_burst), np.quantile(out_burst, 0.25), np.quantile(out_burst, 0.75), skew(out_burst), \
               np.min(in_burst), np.max(in_burst), np.mean(in_burst), np.median(in_burst), np.std(in_burst), np.var(
            in_burst), np.quantile(in_burst, 0.25), np.quantile(in_burst, 0.75), skew(in_burst)

    def getCellPreposition(self):

        idx_list = []
        start = True

        cells = self.total_cell[:300]
        for c in range(len(cells) - 1):
            if cells[0] == 1:
                start = True
            else:
                start = False
            if cells[c] != cells[c + 1]:
                idx_list.append(c + 1)

        if idx_list[-1] != len(cells):
            idx_list.append(len(cells))

        ori = [idx_list[0]] + np.diff(idx_list).tolist()
        if start is True:
            in_pre = ori[0::2]
        else:
            in_pre = ori[1::2]

        return np.sum(in_pre), len(in_pre)

    def getCellOrdering(self):
        c1 = 0
        c2 = 0
        temp1 = []
        temp2 = []

        for cell in self.total_cell:
            if cell == 1:
                temp1.append(c1)
            c1 += 1
            if cell == -1:
                temp2.append(c2)
            c2 += 1

        return np.max(temp2), np.mean(temp2), np.median(temp2), np.std(temp2), np.var(temp2), np.quantile(temp2,
                                                                                                          0.25), np.quantile(
            temp2, 0.75), skew(temp2), \
               np.max(temp1), np.mean(temp1), np.median(temp1), np.std(temp1), np.var(temp1), np.quantile(temp1,
                                                                                                          0.25), np.quantile(
            temp1, 0.75), skew(temp1)

    def getCellConcentration(self):

        out_concentration = []

        chunks = [self.total_cell[x:x + 30] for x in range(0, len(self.total_cell), 30)]
        for item in chunks:
            o = 0
            for p in item:
                if p == 1:
                    o += 1
            out_concentration.append(o)

        return np.max(out_concentration), np.mean(out_concentration), np.median(out_concentration), np.std(
            out_concentration), np.var(out_concentration), np.quantile(out_concentration, 0.25), np.quantile(
            out_concentration, 0.75), skew(out_concentration)

    def getFirstLast30Cell(self):

        f30In = []
        f30Out = []
        l30In = []
        l30Out = []

        for cell in self.total_cell[:30]:
            if cell == 1:
                f30Out.append(cell)
            else:
                f30In.append(cell)

        for cell in self.total_cell[-30:]:
            if cell == 1:
                l30Out.append(cell)
            else:
                l30In.append(cell)

        percf30In = len(f30In) / float(30)
        percf30Out = len(f30Out) / float(30)
        percl30In = len(l30In) / float(30)
        percl30Out = len(l30Out) / float(30)

        return len(f30In), len(f30Out), len(l30In), len(l30Out), percf30In, percf30Out, percl30In, percl30Out

    def getGeneralInformation(self):

        num_of_total_pkts = len(self.total_cell)
        num_of_out_pkts = len(np.array(self.total_cell) == 1)
        num_of_in_pkts = len(np.array(self.total_cell) == -1)

        perc_out = num_of_out_pkts / float(num_of_total_pkts)
        perc_in = num_of_in_pkts / float(num_of_total_pkts)

        duration = self.total_time_list[-1] - self.total_time_list[0]

        pkt_per_sec = num_of_total_pkts / duration
        out_pkt_per_sec = num_of_out_pkts / duration
        in_pkt_per_sec = num_of_in_pkts / duration

        bytes_per_sec = np.sum(list(map(abs, self.total_cell))) / duration
        out_bytes_per_sec = np.sum(np.array(self.total_cell) == 1) / duration
        in_bytes_per_sec = abs(np.sum(np.array(self.total_cell) == -1)) / duration

        return num_of_total_pkts, num_of_out_pkts, num_of_in_pkts, duration, pkt_per_sec, out_pkt_per_sec, in_pkt_per_sec, bytes_per_sec, out_bytes_per_sec, in_bytes_per_sec

    def getFeature(self):

        mean_C, std_C, var_C, skew_C = self.getCellLength()

        minIat, maxIat, meanIat, medianIat, stdIat, varIat, fquanIat, lquanIat, skewIat, minOIat, maxOIat, meanOIat, medianOIat, stdOIat, varOIat, fquanOIat, lquanOIat, skewOIat, minIIat, maxIIat, meanIIat, medianIIat, stdIIat, varIIat, fquanIIat, lquanIIat, skewIIat = self.getCellIAT()

        max_B, mean_B, median_B, std_B, var_B, fquan_B, lquan_B, skew_B, \
        max_OB, mean_OB, median_OB, std_OB, var_OB, fquan_OB, lquan_OB, skew_OB, \
        max_IB, mean_IB, median_IB, std_IB, var_IB, fquan_IB, lquan_IB, skew_IB = self.getCellBurst()  # len_IB, var_IB len_OB, len_B

        # f30_len_B
        f30_max_B, f30_mean_B, f30_median_B, f30_std_B, f30_var_B, f30_fquan_B, f30_lquan_B, f30_skew_B, \
        f30_min_OB, f30_max_OB, f30_mean_OB, f30_median_OB, f30_std_OB, f30_var_OB, f30_fquan_OB, f30_lquan_OB, f30_skew_OB, \
        f30_min_IB, f30_max_IB, f30_mean_IB, f30_median_IB, f30_std_IB, f30_var_IB, f30_fquan_IB, f30_lquan_IB, f30_skew_IB = self.get30CellBurst()

        incoming_preposition, len_incoming_preposition = self.getCellPreposition()

        max_in_o, mean_in_o, median_in_o, std_in_o, var_in_o, fquan_in_o, lquan_in_o, skew_in_o, \
        max_out_o, mean_out_o, median_out_o, std_out_o, var_out_o, fquan_out_o, lquan_out_o, skew_out_o = self.getCellOrdering()

        max_O_conc, mean_O_conc, median_O_conc, std_O_conc, var_O_conc, fquan_O_conc, lauqn_O_conc, skew_O_conc = self.getCellConcentration()  # len_O_conc

        f30In, f30Out, l30In, l30Out, percf30In, percf30Out, percl30In, percl30Out = self.getFirstLast30Cell()

        ntp, nop, nip, dura, pps, opps, ipps, bps, obps, ibps = self.getGeneralInformation()  # nip, pco

        feature_array = [
            mean_C, std_C, var_C, skew_C,  # 0, 1, 2, 3

            minIat, maxIat, meanIat, medianIat, stdIat, varIat, fquanIat, lquanIat, skewIat,  # 4-12
            minOIat, maxOIat, meanOIat, medianOIat, stdOIat, varOIat, fquanOIat, lquanOIat, skewOIat,  # 13-21
            minIIat, maxIIat, meanIIat, medianIIat, stdIIat, varIIat, fquanIIat, lquanIIat, skewIIat,  # 22-30

            max_B, mean_B, median_B, std_B, var_B, fquan_B, lquan_B, skew_B,  # 31-38
            max_OB, mean_OB, median_OB, std_OB, var_OB, fquan_OB, lquan_OB, skew_OB,  # 39-46
            max_IB, mean_IB, median_IB, std_IB, var_IB, fquan_IB, lquan_IB, skew_IB,  # 47-54

            f30_max_B, f30_mean_B, f30_median_B, f30_std_B, f30_var_B, f30_fquan_B, f30_lquan_B, f30_skew_B,  # 55-62
            f30_min_OB, f30_max_OB, f30_mean_OB, f30_median_OB, f30_std_OB, f30_var_OB, f30_fquan_OB, f30_lquan_OB,
            f30_skew_OB,  # 63-71
            f30_min_IB, f30_max_IB, f30_mean_IB, f30_median_IB, f30_std_IB, f30_var_IB, f30_fquan_IB, f30_lquan_IB,
            f30_skew_IB,  # 72-80

            incoming_preposition, len_incoming_preposition,  # 81-82

            max_in_o, mean_in_o, median_in_o, std_in_o, var_in_o, fquan_in_o, lquan_in_o, skew_in_o,  # 83-90
            max_out_o, mean_out_o, median_out_o, std_out_o, var_out_o, fquan_out_o, lquan_out_o, skew_out_o,  # 91-98

            max_O_conc, mean_O_conc, median_O_conc, std_O_conc, var_O_conc, fquan_O_conc, lauqn_O_conc, skew_O_conc,
            # 99-106

            f30In, f30Out, l30In, l30Out, percf30In, percf30Out, percl30In, percl30Out,  # 107-114

            ntp, nop, nip, dura, pps, opps, ipps, bps, obps, ibps  # 115-124
        ]

        return feature_array