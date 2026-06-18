"""
Core simulation constants and configuration parameters.

Note: These values are the production configuration that was in use.
The values here are from the actively-used configuration.
"""

# Traffic Generator parameters

lambda_0 = 20                                                                           # Base connection arrival rate
MEAN_HOLDING_TIME = 20                                                                  # Average connection duration

# Traffic growth rates
alpha =  5                                                                              # traffic growth %
delta = 2.5                                                                             # traffic growth % (Higher rate for stress testing)
Traffic_growth_days = 30

# Fiber parameters
Num_cores =3
FIBERS = 3
C_BAND_SLOTS = 320
TOTAL_SLOTS = 960                                                                      # Routing and Spectrum slot calculation parameters
k_paths: int = 3
DELTA = 12.5 * 10 ** 9
N_TRX = 3
SPECTRAL_EFFICIENCY = [2, 4, 5, 6]                                                      # spectral efficiency of MF of 1-4, unit is b/s/Hz BPSK. QPSK, 8QAM, 16QAM
DATARATE = [100]
MOR_BY_BAND = {
    # C-only column
    "C":      [9900, 2700, 1300, 700],                                                  # [QPSK, 16QAM, 32QAM, 64QAM]

    # C+L band -> use 2 *the C (first) column under C+L (effectively slightly smaller reach)
    "C+L_C":  [9900, 2400, 1300, 700],

    # C+L band -> use the L column (second) under C+L
    "C+L_L":  [8400, 2200, 1100, 600],
}
NUMBER_OF_MFs =4

# Upgrade Parameters
Upgrade_initiation_days = 180

# Upgrade thresholds
blocked_connection_prob_threshold_need_for_upgrade = 0.01  #                             Higher threshold for triggering upgrades
blocked_connection_prob_threshold_plan_checker =  0.02
PER_SD_BLOCK_THRESHOLD =  0.02

MIN_SD_ARRIVALS = 30
MIN_PATH_ARRIVALS = 15

# Simulation analysis and stoppage parameters

Total_number_of_allowed_blocked_probability_in_network_lifetime = 0.1
WARMUP = 1000


# Cost Paramteers

d = 0.2                       # Yearly equipment Depreciation value (20%)
I = 0.03                      # Yearly inflation value (3%)
alpha_c= 18000                # C_BAND_EDFA (USD per unit)
alpha_l= 21000                # L_BAND_EDFA (USD per unit)
mu= 4500                      # DEMUX for L-band (USD per unit)
w= 4500                       # MUX for L-band (USD per unit)
w_c= 20000                    # C_BAND_WSS (USD per unit)
w_l = 23500                   # L_BAND_WSS (USD per unit)
beta_c = 4500                 # C_BAND_TRANSCEIVER (400G) (USD per unit)
beta_l= 7200                  # L_BAND_TRANSCEIVER (400G) (USD per unit)
f_l= 500                      # FIBER_LIGHTING_COST_PER_KM (USD per Km)
C_3c= 1250                    # THREE_CORE_FIBER_PURCHASE_COST_PER_KM
K_d= 600                      # THREE_CORE_FIBER_DEPLOYMENT_PER_KM
L_s= 100                      # Length of the span in Kms
m= 6                          # Interval between upgrade initiation (6 months)
tau= 30                       # Time slot for updation of traffic (30 days)


# OPEX (operating expenditures)

O_C1 =  0.41                  # single-core, single-band (per time slot) (USD per Km per day)
O_CL1 = 0.66                  # single-core, dual-band  (USD per Km per day)
O_C3 =  0.9                   # three-core, single-band (USD per Km per day)
O_CL3 =  1.44                 # three-core, dual-band (USD per Km per day)

# Workforce costs

W_fc  = 50000                #(USD per job)
W_fcl = 90000                #(USD per job)
W_b   = 35000                #(USD per job)
W_c  =  75000                #(USD per job)

# Time to perform different upgrade actions

t_C1 = 0                     # in days
t_CL1 = 15                   # in days
t_b = 23                     # in days
t_3C = 70                    # in days

# Budget for different upgrade

b_C1 = 120000             # Budget For 100 Km route (Includes Equipment + Lighting + Workforce)
b_CL1 = 220000            # Budget For 100 Km route (Dual-band equipment + Workforce)
b_ba = 95000              # Budget For 100 Km route (L-band add-on equipment + Workforce)
b_3c = 260000             # Budget For 100 Km route (MCF cable + deployment + Workforce)


from dataclasses import dataclass
from typing import List

@dataclass
class ConnectionData:
    path: List[int]
    link_ids: List[int]
    fs_index: int
    slice_window_size: int
    mf_index: int
    arrival_time: float
    holding_time: float
    departure_time: float
    datarate: float
    fibers_used: List[int]
    cores_used: List[int]

@dataclass
class ConnectionData_need_upgrade:
    path_nu: List[int]
    link_ids_nu: List[int]
    fs_nu: int
    sw_nu: int
    mf_nu: int
    arrival_time_nu: float
    holding_time_nu: float
    departure_time_nu: float
    rate_nu: float
    fibers_used_nu: List[int]
    cores_used_nu: List[int]

@dataclass
class ConnectionData_plan_checker:
    path_pc: List[int]
    link_ids_pc: List[int]
    fs_pc: int
    sw_pc: int
    mf_pc: int
    arrival_time_pc: float
    holding_time_pc: float
    departure_time_pc: float
    rate_pc: float
    fibers_used_pc: List[int]
    cores_used_pc: List[int]
