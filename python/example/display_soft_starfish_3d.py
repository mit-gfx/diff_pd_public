import sys
sys.path.append('../')
import os

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error, print_warning
from py_diff_pd.common.project_path import root_path

if __name__ == '__main__':
    from soft_starfish_3d import load_csv_data
    measurement_data, info = load_csv_data('soft_starfish_3d/data_horizontal_cyclic1.csv')

    t = measurement_data['time']
    m1x = measurement_data['M1_rel_x']
    m1z = measurement_data['M1_rel_z']
    m4x = measurement_data['M4_rel_x']
    m4z = measurement_data['M4_rel_z']
    dl = measurement_data['dl']

    # Plot the optimization progress.
    plt.rc('pdf', fonttype=42)
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)

    fig = plt.figure(figsize=(12, 8))
    ax_control = fig.add_subplot(311)
    ax_loc_x = fig.add_subplot(312)
    ax_loc_z = fig.add_subplot(313)

    ax_control.set_position((0.1, 0.76, 0.8, 0.23))
    ax_control.plot(t, dl * 1000, color='tab:blue')
    ax_control.set_ylim([-5, 25])
    ax_control.set_xlabel('time (s)')
    ax_control.set_ylabel('contraction (mm)')
    ax_control.grid(True, which='both')

    ax_loc_x.set_position((0.1, 0.43, 0.8, 0.23))
    ax_loc_x.plot(t, m1x * 1000, color='tab:red', label='marker 1')
    ax_loc_x.plot(t, m4x * 1000, color='tab:green', label='marker 4')
    ax_loc_x.set_xlabel('time (s)')
    ax_loc_x.set_ylabel('x loc (mm)')
    ax_loc_x.legend()
    ax_loc_x.grid(True, which='both')

    ax_loc_z.set_position((0.1, 0.1, 0.8, 0.23))
    ax_loc_z.plot(t, m1z * 1000, color='tab:red', label='marker 1')
    ax_loc_z.plot(t, m4z * 1000, color='tab:green', label='marker 4')
    ax_loc_z.set_ylim([-25, 60])
    ax_loc_z.set_xlabel('time (s)')
    ax_loc_z.set_ylabel('y loc (mm)')
    ax_loc_z.legend()
    ax_loc_z.grid(True, which='both')

    plt.show()
    fig.savefig('soft_starfish_3d/cyclic.pdf')

    #################################################
    # Air test.
    #################################################
    def get_com(file_name):
        measurement_data, info = load_csv_data('soft_starfish_3d/{}'.format(file_name), check=False)

        t = measurement_data['time']
        m1x = measurement_data['M1_rel_x']
        m1z = measurement_data['M1_rel_z']
        m2x = measurement_data['M2_rel_x']
        m2z = measurement_data['M2_rel_z']
        m3x = measurement_data['M3_rel_x']
        m3z = measurement_data['M3_rel_z']
        m4x = measurement_data['M4_rel_x']
        m4z = measurement_data['M4_rel_z']
        mx = (m1x + m2x + m3x + m4x) / 4
        mz = (m1z + m2z + m3z + m4z) / 4
        return mx, mz, measurement_data
    mx_water, mz_water, measurement_water = get_com('data_horizontal_cyclic3.csv')
    mx_air, mz_air, measurement_air = get_com('data_airtest.csv')
    t_water = measurement_water['time']
    t_air = measurement_air['time']

    # Plot the optimization progress.
    plt.rc('pdf', fonttype=42)
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)

    fig = plt.figure(figsize=(12, 10))
    ax_x = fig.add_subplot(211)
    ax_x.plot(t_water, mx_water * 1000, label='water test', color='tab:cyan')
    ax_x.plot(t_air, mx_air * 1000, label='air test', color='tab:orange')
    ax_x.grid(True, which='both')
    ax_x.set_xlabel('time (s)')
    ax_x.set_ylabel('x position (mm)')
    ax_x.legend()

    ax_y = fig.add_subplot(212)
    ax_y.plot(t_water, mz_water * 1000, label='water test', color='tab:cyan')
    ax_y.plot(t_air, mz_air * 1000, label='air test', color='tab:orange')
    ax_y.grid(True, which='both')
    ax_y.set_xlabel('time (s)')
    ax_y.set_ylabel('y position (mm)')
    ax_y.legend()

    plt.show()
    fig.savefig('soft_starfish_3d/air.png')

    ##############################
    # Real experiments.
    ##############################
    # Plot the optimization progress.
    plt.rc('pdf', fonttype=42)
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)

    mx_baseline, _, measurement_baseline = get_com('data_baseline.csv')
    mx_round2, _, measurement_round2 = get_com('data_horizontal_cyclic2.csv')
    mx_round3, _, measurement_round3 = get_com('data_horizontal_cyclic3.csv')
    t_baseline = measurement_baseline['time']
    t_round2 = measurement_round2['time']
    t_round3 = measurement_round3['time']

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.plot(t_baseline, -mx_baseline * 1000, label='round 0 (baseline)', color='tab:red')
    ax.plot(t_round2, -mx_round2 * 1000, label='round 1', color='tab:green')
    ax.plot(t_round3, -mx_round3 * 1000, label='round 2', color='tab:blue')
    ax.grid(True, which='both')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('distance traveled (mm)')
    ax.legend()

    plt.show()
    fig.savefig('soft_starfish_3d/experiment_data.pdf')
