import os
import sys
import glob
import tqdm
import numpy as np

from scapy.all import rdpcap


FOLDER_PATH = sys.argv[1]
LOCAL_IP = ['192.168.160.100', '192.168.160.101', '192.168.160.102', '192.168.160.103', '192.168.160.104', '192.168.160.105', '192.168.160.106', '192.168.160.107', '192.168.160.108', '192.168.160.109', '192.168.160.110']


def removeInstance(instance):

    new_instance = dict()
    instance = dict(sorted(instance.items(), key=(lambda x: x[1])))
    q1 = np.quantile(list(instance.values()), 0.25)
    q3 = np.quantile(list(instance.values()), 0.75)

    for key, value in instance.items():
        if q1 - 1.5 * (q3 - q1) < value and value < q3 + 1.5 * (q3 - q1):
            new_instance[key] = value
    new_instance = dict(sorted(new_instance.items(), key=(lambda x: x[0])))

    return new_instance


def Main():

    website_list = sorted(glob.glob(FOLDER_PATH + '/**/'))

    for website in tqdm.tqdm(website_list):
        pcap_list = glob.glob(website + '/*.pcap')
        instance = dict()
        for cap in pcap_list:
            if cap not in instance:
                instance[cap] = 0
            sum = 0
            try:
                pcap = rdpcap(cap)
            except:
                continue
            for pkt in pcap:
                try:
                    # Remove C2 server packets
                    if pkt['IP'].src == '192.168.160.10':
                        continue
                    if pkt['IP'].dst == '192.168.160.10':
                        continue
                    # We gather only incoming packets
                    if pkt['IP'].src in LOCAL_IP:
                        continue

                    sum += len(pkt)
                except:
                    continue
            instance[cap] = sum

        new_instance = removeInstance(instance)
        fd = open(os.path.join(sys.argv[2], '%s.txt' % website.split('/')[-2]), 'w')
        for key, value in new_instance.items():
            fd.write(key)
            fd.write('\n')
        fd.close()


if __name__ == '__main__':

    Main()
