import cmath
import pyshark
import numpy as np
import math
from textwrap import wrap
from scipy.io import savemat

def hex2dec(hex_value):
    hex2array = np.array(wrap(hex_value, 2))
    array_joined = "".join(hex2array)
    decimal_value = int(array_joined, 16)
    return decimal_value

def flip_hex(hex_value):
    hex2array = np.array(wrap(hex_value, 2))
    hex2array_flipped = np.flip(hex2array)
    hex2array_flipped = "".join(hex2array_flipped)
    return hex2array_flipped

def bfi_angles(bin_chunk, LSB, NSUBC_VALID, order_bits):
    bfi_angles_all = []
    for l in range(NSUBC_VALID):
        chunk = bin_chunk[l]
        if len(chunk) < sum(order_bits):
            continue
        idx = 0
        bfi_angles_single = np.zeros(len(order_bits), dtype=int)
        for k in range(len(order_bits)):
            angle_bin = chunk[idx:idx + order_bits[k]]
            bfi_angles_single[k] = sum([int(e) * 2 ** j for j, e in enumerate(angle_bin)])
            if LSB:
                # LSB-first already handled by reversing each byte; here angle bits are in natural order
                pass
            idx += order_bits[k]
        bfi_angles_all.append(bfi_angles_single)
    bfi_angles_all = np.array(bfi_angles_all)
    return bfi_angles_all

def vmatrices(angle, phi_bit, psi_bit, NSUBC_VALID, Nr, Nc_users, config):
    if config == "1x1":
        const1_phi = 1 / 2 ** (phi_bit - 1)
        const2_phi = 1 / 2 ** (phi_bit)

        phi_11 = math.pi * (const2_phi + const1_phi * angle[:, 0])
        psi_21 = math.pi * (const2_phi + const1_phi * angle[:, 1])

        v_matrix_all = []
        for s in range(NSUBC_VALID):
            if s >= angle.shape[0]:
                break
            D_1 = [[cmath.exp(1j * phi_11[s]), 0],
                   [0, 1]]

            G_21 = [[math.cos(psi_21[s]), math.sin(psi_21[s])],
                    [-math.sin(psi_21[s]), math.cos(psi_21[s])]]

            I_matrix = np.eye(2)
            V = np.matmul(np.matmul(D_1, np.transpose(G_21)), I_matrix)
            v_matrix = np.transpose(V)
            v_matrix_all.append(v_matrix)

    v_matrix_all = np.stack(v_matrix_all, axis=1)
    v_matrix_all = np.moveaxis(v_matrix_all, [1, 2, 0], [0, 1, 2])
    return v_matrix_all

file_name = '8_21_1_1.pcapng'
standard = 'AC'   # 'AC' or 'AX'
mimo = 'SU'       # 'SU' or 'MU' (AX MU not supported in this script)
config = '1x1'    
bw = 80           # 20/40/80/160
# MAC = '98:25:4a:fe:54:b5'
# MAC = '98:25:4a:fe:5b:96'
num_packet_to_process = 20000
saved_vmatrices = 'V_8_21_1_1'
saved_angles = 'bfa_8_21_1_1'
saved_timestamps = 'timestamps_8_21_1_1'
mat_file = 'multi_8_21_1_1.mat'

LSB = True

if __name__ == '__main__':
    file_name = 'traces/' + file_name
    standard = standard
    mimo = mimo
    config = config
    bw = int(bw)
    num_packet_to_process = int(num_packet_to_process)
    saved_vmatrices = 'vmatrix/' + saved_vmatrices
    saved_angles = 'bfa/' + saved_angles
    saved_timestamps = 'vmatrix/' + saved_timestamps
    mat_file = 'mat/' + mat_file

    if mimo == "MU" and standard == "AX":
        print("mu-mimo is not available for AX yet, we will add this feature soon")
    else:
        print("Processing")

    if standard == "AC":
        if bw == 80:
            subcarrier_idxs = np.arange(-122, 123)
            pilot_n_null = np.array([-104, -76, -40, -12, -1, 0, 1, 10, 38, 74, 102])
            subcarrier_idxs = np.setdiff1d(subcarrier_idxs, pilot_n_null)
        elif bw == 40:
            subcarrier_idxs = np.arange(-58, 59)
            pilot_n_null = np.array([-54, -26, -12, -1, 0, 1, 10, 24, 52])
            subcarrier_idxs = np.setdiff1d(subcarrier_idxs, pilot_n_null)
        elif bw == 20:
            subcarrier_idxs = np.arange(-28, 29)
            pilot_n_null = np.array([-21, -8, 0, 6, 21])
            subcarrier_idxs = np.setdiff1d(subcarrier_idxs, pilot_n_null)
        else:
            print("input a valid bandwidth for IEEE 802.11ac")

    if standard == "AX":
        if bw == 160:
            subcarrier_idxs = np.arange(-1012, 1013, 4)
            pilot_n_null = np.array([-512, -8, -4, 0, 4, 8, 512])
            subcarrier_idxs = np.setdiff1d(subcarrier_idxs, pilot_n_null)
        elif bw == 80:
            subcarrier_idxs = np.arange(-500, 504, 4)
            pilot_n_null = np.array([0])
            subcarrier_idxs = np.setdiff1d(subcarrier_idxs, pilot_n_null)
        elif bw == 40:
            subcarrier_idxs = np.arange(-244, 248, 4)
            pilot_n_null = np.array([0])
            subcarrier_idxs = np.setdiff1d(subcarrier_idxs, pilot_n_null)
        elif bw == 20:
            subcarrier_idxs = np.arange(-122, 126, 4)
            pilot_n_null = np.array([0])
            subcarrier_idxs = np.setdiff1d(subcarrier_idxs, pilot_n_null)
        else:
            print("input a valid bandwidth for IEEE 802.11ac")

    try:
        cap = pyshark.FileCapture(file_name, use_json=True, include_raw=True)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    bfi_angles_all_packets = []
    v_matrices_all = []

    # timestamps aligned to V frames
    timestamps = []

    try:
        for p, pkt in enumerate(cap):
            if p >= num_packet_to_process:
                break
            packet = pkt.frame_raw.value
            # real timestamp (epoch seconds)
            try:
                ts = float(pkt.sniff_timestamp)
            except Exception:
                try:
                    ts = float(pkt.frame_info.time_epoch)
                except Exception:
                    ts = float('nan')
            print('packet___________ ' + str(p) + '\n\n\n')

            Header_rivision_dec = hex2dec(flip_hex(packet[0:2]))
            Header_pad_dec = hex2dec(flip_hex(packet[2:4]))
            Header_length_dec = hex2dec(flip_hex(packet[4:8]))
            i = Header_length_dec * 2

            Frame_Control_Field_hex = packet[i:(i + 4)]
            packet_duration = packet[(i + 4):(i + 8)]
            packet_destination_mac = packet[(i + 8):(i + 20)]
            packet_sender_mac = packet[(i + 20):(i + 32)]
            packet_BSS_ID = packet[(i + 32):(i + 44)]
            packet_sequence_number = packet[(i + 44):(i + 48)]
            packet_HE_category = packet[(i + 48):(i + 50)]
            packet_CQI = packet[(i + 50):(i + 52)]

            if standard == "AX":
                packet_mimo_control = packet[(i + 52):(i + 58)]
                packet_mimo_control_binary = ''.join(format(int(char, 16), '04b') for char in flip_hex(packet_mimo_control))
                codebook_info = packet_mimo_control_binary[13]
                packet_snr = packet[(i + 58):(i + 58 + 2 * int(config[-1]))]
                frame_check_sequence = packet[-8:]

            if standard == "AC":
                packet_mimo_control = packet[(i + 52):(i + 58)]
                packet_mimo_control_binary = ''.join(format(int(char, 16), '04b') for char in flip_hex(packet_mimo_control))
                codebook_info = packet_mimo_control_binary[13]
                packet_snr = packet[(i + 58):(i + 58 + 2 * int(config[-1]))]
                frame_check_sequence = packet[-8:]

            if mimo == "SU":
                if codebook_info == "1":
                    psi_bit = 4
                    phi_bit = psi_bit + 2
                else:
                    psi_bit = 2
                    phi_bit = psi_bit + 2
            elif mimo == "MU":
                if codebook_info == "1":
                    psi_bit = 7
                    phi_bit = psi_bit + 2
                else:
                    psi_bit = 5
                    phi_bit = phi_bit + 2

            if config == "1x1":
                Nc_users = 1
                Nr = 1
                phi_numbers = 1
                psi_numbers = 1
                order_angles = ['phi_11', 'psi_21']
                order_bits = [phi_bit, psi_bit]
                tot_angles_users = phi_numbers + psi_numbers
                tot_bits_users = phi_numbers * phi_bit + psi_numbers * psi_bit

            else:
                print("the antenna configuration that you have is not supported right now, you will update other configurations soon, stay tuned")

            NSUBC_VALID = 232  
            length_angles_users_bits = NSUBC_VALID * tot_bits_users
            length_angles_users = math.floor(length_angles_users_bits / 8)

            if standard == "AX":
                Feedback_angles = packet[(i + 62 + 2 * int(config[-1])):(len(packet) - 8)]
                Feedback_angles_splitted = np.array(wrap(Feedback_angles, 2))
                Feedback_angles_bin = ""
            if standard == "AC":
                Feedback_angles = packet[(i + 58 + 2 * int(config[-1])):(len(packet) - 8)]
                Feedback_angles_splitted = np.array(wrap(Feedback_angles, 2))
                Feedback_angles_bin = ""

            for i in range(0, len(Feedback_angles_splitted)):
                bin_str = str(format(hex2dec(Feedback_angles_splitted[i]), '08b'))
                if LSB:
                    bin_str = bin_str[::-1]
                Feedback_angles_bin += bin_str

            Feed_back_angles_bin_chunk = np.array(wrap(Feedback_angles_bin[:(tot_bits_users * NSUBC_VALID)], tot_bits_users))

            angle = bfi_angles(Feed_back_angles_bin_chunk, LSB, NSUBC_VALID, order_bits)
            print(f"Packet {p}: Angle shape: {angle.shape}, NSUBC_VALID: {NSUBC_VALID}")
            v_matrices_all.append(vmatrices(angle, phi_bit, psi_bit, NSUBC_VALID, Nr, Nc_users, config))
            bfi_angles_all_packets.append(angle)
            timestamps.append(ts)
    except StopIteration:
        print("No more packets to process or file is empty")
    finally:
        try:
            cap.close()
        except Exception:
            pass

    np.save(saved_vmatrices, v_matrices_all)
    np.save(saved_angles, bfi_angles_all_packets)
    np.save(saved_timestamps, np.asarray(timestamps, dtype=np.float64))
    savemat(mat_file, {'bfi_angles': bfi_angles_all_packets})

    print(f"Saved V, angles, timestamps to {saved_vmatrices}, {saved_angles}, {saved_timestamps}; MAT to {mat_file}")









