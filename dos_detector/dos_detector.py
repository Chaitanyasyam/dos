from scapy.all import IP, rdpcap, sniff
from sklearn.svm import OneClassSVM
from decimal import Decimal
import numpy as np
import argparse
import sys


def packet_rate(dataset):
    times = [packet.time for packet in dataset if IP in packet]
    if len(times) < 2:
        return Decimal(0)
    return Decimal(len(times)) / Decimal(times[-1] - times[0])


def captured_packets(captured_set):
    packets_by_srcip = {}
    for packet in captured_set:
        if IP in packet:
            srcip = packet[IP].src
            if srcip not in packets_by_srcip:
                packets_by_srcip[srcip] = []
            packets_by_srcip[srcip].append(packet)

    packet_rates = []
    for packets in packets_by_srcip.values():
        if len(packets) > 1:
            times = [packet.time for packet in packets]
            rate = Decimal(len(times)) / Decimal(times[-1] - times[0])
            packet_rates.append(rate)

    if packet_rates:
        return np.mean(packet_rates)
    else:
        return 0


def main(pcap_file):
    print("[+] LOADING TRAINING DATASET...")

    try:
        valid = rdpcap(pcap_file)
    except FileNotFoundError:
        print(f"[-] File {pcap_file} not found")
        sys.exit(1)

    valid_rate = packet_rate(valid)

    if valid_rate == 0:
        print("[-] Not enough packets to calculate rate.")
        sys.exit(1)

    print(f"[+] Training Packet Rate: {valid_rate:.4f} pkt/s")

    model = OneClassSVM(kernel='rbf', nu=0.1, gamma=0.1)
    model.fit(np.array([[float(valid_rate)]]))

    print("[+] MONITORING START...")

    try:
        while True:
            captured_traffic = sniff(count=5000, timeout=10)  # 10 seconds sniffing

            print("[+] ANALYSIS START...")
            capt_rate = captured_packets(captured_traffic)
            threshold = float(valid_rate * Decimal(1.5))

            is_outlier = model.predict(np.array([[float(capt_rate)]]))[0] == -1

            if capt_rate > threshold and is_outlier:
                prob = min(100, ((capt_rate - valid_rate) / valid_rate) * 100)
            else:
                prob = 0

            print(f"[!] Probability of DoS attack: {prob:.2f}%")
            print(f"[+] Captured Rate: {capt_rate:.4f} pkt/s\n")

    except KeyboardInterrupt:
        print("[+] Quitting the program...")
        sys.exit(0)
    except Exception as e:
        print(f"[-] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DoS Attack Detection Script using Scapy and OneClassSVM")
    parser.add_argument("-i", "--input", required=True, help="Input PCAP file for training dataset")

    args = parser.parse_args()
    main(args.input)