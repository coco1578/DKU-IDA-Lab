import os
import glob

from TorNetwork.IDAPreprocess.packet_parser import flowGenerator


IP = [
    '10.178.0.2',
    '10.170.0.5',
    '10.148.0.2',
    '10.152.0.2',
    '10.140.0.2',
    '10.146.0.4',
    '10.128.0.2',
    '10.150.0.2',
    '10.138.0.2',
    '10.142.0.2',
    '10.162.0.2',
    '10.158.0.2',
    '10.132.0.2',
    '10.166.0.2',
    '10.156.0.2',
    '10.154.0.2',
    '10.164.0.2',
    '10.172.0.2'
]


def main():

    base_folder_path = glob.glob(r'C:\Users\coco\Downloads\0625 (1)' + r'\**')

    for base_folder in base_folder_path:
        class_label = dict()
        class_idx = 0

        fd = open(os.path.join(r'C:\Users\coco\Desktop', os.path.split(base_folder)[1] + '.spa'), 'w')
        folder_path = sorted(glob.glob(base_folder + r'\**'))
        for folder in folder_path:

            folder_name = os.path.split(folder)[1]

            if folder_name not in class_label:
                class_label[folder_name] = class_idx

            pcaps = glob.glob(folder + r'\*.pcap')
            for pcap in pcaps:
                fw = flowGenerator(pcap, IP)
                try:
                    features = fw.getFeature()
                    fd.write('%d ' % class_idx)

                    array = list()
                    for i in range(len(features)):
                        array.append('%d:%f' % (i+1, features[i]))

                    fd.write(' '.join(array))
                    fd.write('\n')

                except:
                    continue

            class_idx += 1
        fd.close()

if __name__ == '__main__':

    main()