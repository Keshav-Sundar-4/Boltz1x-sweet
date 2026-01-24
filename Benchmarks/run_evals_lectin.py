import argparse
import sys
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import copy
import csv

import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from scipy import stats

from Bio.PDB import PDBParser, MMCIFParser, Superimposer
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from rdkit import Chem
from rdkit.Chem import rdFMCS, rdMolAlign
from rdkit.Geometry import Point3D

from typing import Dict

# Suppress warnings
warnings.simplefilter('ignore', PDBConstructionWarning)
warnings.simplefilter('ignore', UserWarning)

# --- CONSTANTS ---
BOND_DIST = 2.0        
CONTACT_CUTOFF = 5.0    # [cite: 811]
POCKET_CUTOFF = 10.0    
RING_BOND_CUTOFF = 2.0 

# DockQC Scaling Factors [cite: 876, 877]
D_RRMS = 2.5
D_LRMS = 5.0

STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE', 'SEC', 'PYL', 'HOH', 'WAT'
}

MONO_TYPE_MAP: Dict[str, int] = {
    "05L": 0,   "07E": 1,   "0HX": 2,   "0LP": 3,   "0MK": 4,   "0NZ": 5,   "0UB": 6,   "0WK": 7,   "0XY": 8,   "0YT": 9,   "12E": 10,  "145": 11,  "147": 12,  "149": 13,  "14T": 14,  "15L": 15,  "16F": 16,  "16G": 17,  "16O": 18,  "17T": 19,  "18D": 20,  "18O": 21,  "1CF": 22,  "1GL": 23,  "1GN": 24,  "1S3": 25,  "1S4": 26,  "1SD": 27,  "1X4": 28,  "20S": 29,  "20X": 30,
    "22O": 31,  "22S": 32,  "23V": 33,  "24S": 34,  "25E": 35,  "26O": 36,  "27C": 37,  "289": 38,  "291": 39,  "293": 40,  "2DG": 41,  "2DR": 42,  "2F8": 43,  "2FG": 44,  "2FL": 45,  "2GL": 46,  "2GS": 47,  "2H5": 48,  "2M5": 49,  "2M8": 50,  "2WP": 51,  "32O": 52,  "34V": 53,  "38J": 54,  "3DO": 55,  "3FM": 56,  "3HD": 57,  "3J3": 58,  "3J4": 59,  "3LJ": 60,  "3MG": 61,
    "3MK": 62,  "3R3": 63,  "3S6": 64,  "3YW": 65,  "42D": 66,  "445": 67,  "44S": 68,  "46Z": 69,  "475": 70,  "491": 71,  "49A": 72,  "49S": 73,  "49T": 74,  "49V": 75,  "4AM": 76,  "4CQ": 77,  "4GL": 78,  "4GP": 79,  "4JA": 80,  "4N2": 81,  "4NN": 82,  "4QY": 83,  "4R1": 84,  "4SG": 85,  "4U0": 86,  "4U1": 87,  "4U2": 88,  "4UZ": 89,  "4V5": 90,  "50A": 91,
    "510": 92,  "51N": 93,  "56N": 94,  "57S": 95,  "5DI": 96,  "5GF": 97,  "5GO": 98,  "5KQ": 99,  "5KV": 100, "5L2": 101, "5L3": 102, "5LS": 103, "5LT": 104, "5N6": 105, "5QP": 106, "5TH": 107, "5TJ": 108, "5TK": 109, "5TM": 110, "604": 111, "61J": 112, "62I": 113, "64K": 114, "66O": 115, "6BG": 116,  "6C2": 117,  "6GB": 118,  "6GP": 119,  "6GR": 120,
    "6K3": 121, "6KH": 122, "6KL": 123, "6KS": 124, "6KU": 125, "6KW": 126, "6LS": 127, "6LW": 128, "6MJ": 129, "6MN": 130, "6PY": 131, "6PZ": 132, "6S2": 133, "6UD": 134, "6Y6": 135, "6YR": 136, "6ZC": 137, "73E": 138, "79J": 139, "7CV": 140, "7D1": 141, "7GP": 142, "7JZ": 143, "7K2": 144, "7K3": 145, "7NU": 146, "83Y": 147, "89Y": 148, "8B7": 149, "8B9": 150, "8EX": 151, "8GA": 152,
    "8GG": 153, "8GP": 154, "8LM": 155, "8LR": 156, "8OQ": 157, "8PK": 158, "8S0": 159, "95Z": 160, "96O": 161, "9AM": 162, "9C1": 163, "9CD": 164, "9GP": 165, "9KJ": 166, "9MR": 167, "9OK": 168, "9PG": 169, "9QG": 170, "9QZ": 171, "9S7": 172, "9SG": 173, "9SJ": 174, "9SM": 175, "9SP": 176, "9T1": 177, "9T7": 178, "9VP": 179, "9WJ": 180, "9WN": 181, "9WZ": 182, "9YW": 183,
    "A0K": 184, "A1Q": 185, "A2G": 186, "A5C": 187, "A6P": 188, "AAL": 189, "ABD": 190, "ABE": 191, "ABF": 192, "ABL": 193, "AC1": 194, "ACR": 195, "ACX": 196, "ADA": 197, "AF1": 198, "AFD": 199, "AFO": 200, "AFP": 201, "AFR": 202, "AGL": 203, "AGR": 204, "AH2": 205, "AH8": 206, "AHG": 207, "AHM": 208, "AHR": 209, "AIG": 210, "ALL": 211, "ALX": 212, "AMG": 213, "AMN": 214, "AMU": 215,
    "AMV": 216, "ANA": 217, "AOG": 218, "AQA": 219, "ARA": 220, "ARB": 221, "ARI": 222, "ARW": 223, "ASC": 224, "ASG": 225, "ASO": 226, "AXP": 227, "AXR": 228, "AY9": 229, "AZC": 230, "B0D": 231, "B16": 232, "B1H": 233, "B1N": 234, "B6D": 235, "B7G": 236, "B8D": 237, "B9D": 238, "BBK": 239, "BBV": 240, "BCD": 241, "BCW": 242, "BDF": 243, "BDG": 244, "BDP": 245, "BDR": 246, "BDZ": 247,
    "BEM": 248, "BFN": 249, "BG6": 250, "BG8": 251, "BGC": 252, "BGL": 253, "BGN": 254, "BGP": 255, "BGS": 256, "BHG": 257, "BM3": 258, "BM7": 259, "BMA": 260, "BMX": 261, "BND": 262, "BNG": 263, "BNX": 264, "BO1": 265, "BOG": 266, "BQY": 267, "BS7": 268, "BTG": 269, "BTU": 270, "BWG": 271, "BXF": 272, "BXX": 273, "BXY": 274, "BZD": 275,
    "C3B": 276, "C3G": 277, "C3X": 278, "C4B": 279, "C4W": 280, "C5X": 281, "CBF": 282, "CBI": 283, "CBK": 284, "CDR": 285, "CE5": 286, "CE6": 287, "CE8": 288, "CEG": 289, "CEX": 290, "CEY": 291, "CEZ": 292, "CGF": 293, "CJB": 294, "CKB": 295, "CKP": 296, "CNP": 297, "CR1": 298, "CR6": 299, "CRA": 300, "CT3": 301, "CTO": 302, "CTR": 303, "CTT": 304,
    "D0N": 305, "D1M": 306, "D5E": 307, "D6G": 308, "DAF": 309, "DAG": 310, "DAN": 311, "DDA": 312, "DDL": 313, "DEG": 314, "DEL": 315, "DFR": 316, "DFX": 317, "DGO": 318, "DGS": 319, "DJB": 320, "DJE": 321, "DK4": 322, "DKX": 323, "DKZ": 324, "DL6": 325, "DLD": 326, "DLF": 327, "DLG": 328, "DO8": 329, "DOM": 330, "DPC": 331, "DQR": 332, "DR2": 333, "DR3": 334, "DR5": 335,
    "DRI": 336, "DSR": 337, "DT6": 338, "DVC": 339, "DYM": 340, "E3M": 341, "E5G": 342, "EAG": 343, "EBG": 344, "EBQ": 345, "EEN": 346, "EEQ": 347, "EGA": 348, "EMP": 349, "EMZ": 350, "EPG": 351, "EQP": 352, "EQV": 353, "ERE": 354, "ERI": 355, "ETT": 356, "F1P": 357, "F1X": 358, "F55": 359, "F58": 360, "F6P": 361, "FBP": 362, "FCA": 363, "FCB": 364, "FCT": 365, "FDP": 366,
    "FDQ": 367, "FFC": 368, "FFX": 369, "FIF": 370, "FK9": 371, "FKD": 372, "FMF": 373, "FMO": 374, "FNG": 375, "FNY": 376, "FRU": 377, "FSA": 378, "FSI": 379, "FSM": 380, "FSR": 381, "FSW": 382, "FUB": 383, "FUC": 384, "FUF": 385, "FUL": 386, "FUY": 387, "FVQ": 388, "FX1": 389, "FYJ": 390, "G0S": 391, "G16": 392, "G1P": 393, "G20": 394, "G28": 395, "G2F": 396,
    "G3F": 397, "G4D": 398, "G4S": 399, "G6D": 400, "G6P": 401, "G6S": 402, "G7P": 403, "G8Z": 404, "GAA": 405, "GAC": 406, "GAD": 407, "GAF": 408, "GAL": 409, "GAT": 410, "GBH": 411, "GC1": 412, "GC4": 413, "GC9": 414, "GCB": 415, "GCD": 416, "GCN": 417, "GCO": 418, "GCS": 419, "GCT": 420, "GCU": 421, "GCV": 422, "GCW": 423, "GDA": 424, "GDL": 425,
    "GE1": 426, "GE3": 427, "GFP": 428, "GIV": 429, "GL0": 430, "GL1": 431, "GL2": 432, "GL4": 433, "GL5": 434, "GL6": 435, "GL7": 436, "GL9": 437, "GLA": 438, "GLC": 439, "GLD": 440, "GLF": 441, "GLG": 442, "GLO": 443, "GLP": 444, "GLS": 445, "GLT": 446, "GM0": 447, "GMB": 448, "GMH": 449, "GMT": 450, "GMZ": 451, "GN1": 452, "GN4": 453, "GNS": 454, "GNX": 455,
    "GP0": 456, "GP1": 457, "GP4": 458, "GPH": 459, "GPK": 460, "GPM": 461, "GPO": 462, "GPQ": 463, "GPU": 464, "GPV": 465, "GPW": 466, "GQ1": 467, "GRF": 468, "GRX": 469, "GS1": 470, "GS9": 471, "GTK": 472, "GTM": 473, "GTR": 474, "GU0": 475, "GU1": 476, "GU2": 477, "GU3": 478, "GU4": 479, "GU5": 480, "GU6": 481, "GU8": 482, "GU9": 483, "GUF": 484, "GUL": 485, "GUP": 486,
    "GUZ": 487, "GXL": 488, "GYE": 489, "GYG": 490, "GYP": 491, "GYU": 492, "GYV": 493, "GZL": 494, "H1M": 495, "H1S": 496, "H2P": 497, "H53": 498, "H6Q": 499, "H6Z": 500, "HBZ": 501, "HD4": 502, "HNV": 503, "HNW": 504, "HSG": 505, "HSH": 506, "HSJ": 507, "HSQ": 508, "HSX": 509, "HSY": 510, "HTG": 511, "HTM": 512, "I57": 513, "IAB": 514, "IDC": 515, "IDF": 516, "IDG": 517, "IDR": 518, 
    "IDS": 519, "IDU": 520, "IDX": 521, "IDY": 522, "IEM": 523, "IN1": 524, "IPT": 525, "ISD": 526, "ISL": 527, "ISX": 528, "IXD": 529, "J5B": 530, "JFZ": 531, "JHM": 532, "JLT": 533, "JS2": 534, "JV4": 535, "JVA": 536, "JVS": 537, "JZR": 538, "K5B": 539, "K99": 540, "KBA": 541, "KBG": 542, "KD5": 543, "KDA": 544, "KDB": 545, "KDD": 546, "KDE": 547, "KDF": 548, "KDM": 549, "KDN": 550, 
    "KDO": 551, "KDR": 552, "KFN": 553, "KG1": 554, "KGM": 555, "KHP": 556, "KME": 557, "KO1": 558, "KO2": 559, "KOT": 560, "KTU": 561,
    "L1L": 562, "L6S": 563, "LAH": 564, "LAK": 565, "LAO": 566, "LAT": 567, "LB2": 568, "LBS": 569, "LBT": 570, "LCN": 571, "LDY": 572, "LEC": 573, "LFR": 574, "LGC": 575, "LGU": 576, "LKA": 577, "LKS": 578, "LNV": 579, "LOG": 580, "LOX": 581, "LRH": 582, "LVO": 583, "LVZ": 584, "LXB": 585, "LXC": 586, "LXZ": 587, "LZ0": 588, "M1F": 589, "M1P": 590, "M2F": 591, "M3N": 592, "M55": 593, "M6D": 594, 
    "M6P": 595, "M7B": 596, "M7P": 597, "M8C": 598, "MA1": 599, "MA2": 600, "MA3": 601, "MA8": 602, "MAF": 603, "MAG": 604, "MAL": 605, "MAN": 606, "MAT": 607, "MAV": 608, "MAW": 609, "MBE": 610, "MBF": 611, "MBG": 612, "MCU": 613, "MDA": 614, "MDP": 615, "MFB": 616, "MFU": 617, "MG5": 618, "MGC": 619, "MGL": 620, "MGS": 621, "MJJ": 622, "MLB": 623, "MLR": 624, "MMA": 625, "MN0": 626, 
    "MNA": 627, "MQG": 628, "MQT": 629, "MRH": 630, "MRP": 631,"MSX": 632, "MTT": 633, "MUB": 634, "MUR": 635, "MVP": 636, "MXY": 637, "MXZ": 638, "MYG": 639, "N1L": 640, "N9S": 641, "NA1": 642, "NAA": 643, "NAG": 644, "NBG": 645, "NBX": 646, "NBY": 647, "NDG": 648, "NFG": 649, "NG1": 650, "NG6": 651, "NGA": 652, "NGC": 653, "NGE": 654, "NGK": 655, "NGR": 656, "NGS": 657, "NGY": 658, "NGZ": 659, "NHF": 660, 
    "NLC": 661, "NM6": 662, "NM9": 663, "NNG": 664, "NPF": 665, "NSQ": 666, "NT1": 667, "NTF": 668, "NTO": 669, "NTP": 670, "NXD": 671, "NYT": 672,
    "O1G": 673, "OAK": 674, "OEL": 675, "OI7": 676, "OPM": 677, "OSU": 678, "OTG": 679, "OTN": 680, "OTU": 681, "OX2": 682, "P53": 683, "P6P": 684, "PA1": 685, "PAV": 686, "PDX": 687, "PH5": 688, "PKM": 689, "PNA": 690, "PNG": 691, "PNJ": 692, "PNW": 693, "PPC": 694, "PRP": 695, "PSG": 696, "PSV": 697, "PUF": 698, "PZU": 699, "QIF": 700, "QKH": 701, "QPS": 702, "R1P": 703, "R1X": 704, "R2B": 705, "R2G": 706,
    "RAE": 707, "RAF": 708, "RAM": 709, "RAO": 710, "RCD": 711, "RER": 712, "RF5": 713, "RGG": 714, "RHA": 715, "RHC": 716, "RI2": 717, "RIB": 718, "RIP": 719, "RM4": 720, "RP3": 721, "RP5": 722, "RP6": 723, "RR7": 724, "RRJ": 725, "RRY": 726, "RST": 727, "RTG": 728, "RTV": 729, "RUG": 730, "RUU": 731, "RV7": 732, "RVG": 733, "RVM": 734, "RWI": 735, "RY7": 736, "RZM": 737, "S7P": 738, "S81": 739,
     "SA0": 740, "SCG": 741, "SCR": 742, "SDY": 743, "SEJ": 744, "SF6": 745, "SF9": 746, 
    "SFJ": 747, "SFU": 748, "SG4": 749, "SG5": 750, "SG6": 751, "SG7": 752, "SGA": 753, "SGC": 754, "SGD": 755, "SGN": 756, "SHB": 757, "SHD": 758, "SHG": 759, "SIA": 760, "SID": 761, "SIO": 762, "SIZ": 763, "SLB": 764, "SLM": 765, "SLT": 766, "SMD": 767, "SN5": 768, "SNG": 769, "SOE": 770, "SOG": 771, 
    "SOR": 772, "SR1": 773, "SSG": 774, "STZ": 775, "SUC": 776, "SUP": 777, "SUS": 778, "SWE": 779, "SZZ": 780, "T68": 781, "T6P": 782, "T6T": 783, "TA6": 784, "TCB": 785, "TCG": 786, "TDG": 787, "TEU": 788, "TF0": 789, "TFU": 790, "TGA": 791, "TGK": 792, "TGR": 793, "TGY": 794, "TH1": 795, "TMR": 796, 
    "TMX": 797, "TNX": 798, "TOA": 799, "TOC": 800, "TQY": 801, "TRE": 802, "TRV": 803, "TS8": 804, "TT7": 805, "TTV": 806, "TTZ": 807, "TU4": 808, "TUG": 809, "TUJ": 810, "TUP": 811, "TUR": 812, "TVD": 813, "TVG": 814, "TVM": 815, "TVS": 816, "TVV": 817, "TVY": 818, "TW7": 819, "TWA": 820, "TWD": 821, "TWG": 822, "TWJ": 823, "TWY": 824, "TXB": 825, "TYV": 826,
    "U1Y": 827, "U2A": 828, "U2D": 829, "U63": 830, "U8V": 831, "U97": 832, "U9A": 833, "U9D": 834, "U9G": 835, "U9J": 836, "U9M": 837, "UAP": 838, "UCD": 839, "UDC": 840, "UEA": 841, "V3M": 842, "V3P": 843, "V71": 844, "VG1": 845, "VTB": 846, "W9T": 847, "WIA": 848, "WOO": 849, "WUN": 850, "X0X": 851, "X1P": 852, "X1X": 853, "X2F": 854, "X6X": 855, "XDX": 856, "XGP": 857, 
    "XIL": 858, "XLF": 859, "XLS": 860, "XMM": 861, "XXM": 862, "XXR": 863, "XXX": 864, "XYF": 865, "XYL": 866, "XYP": 867, "XYS": 868, "XYT": 869, "XYZ": 870, "YIO": 871, "YJM": 872, "YKR": 873, "YO5": 874, "YX0": 875, "YX1": 876, "YYB": 877, "YYH": 878, "YYJ": 879, "YYK": 880, "YYM": 881, "YYQ": 882, "YZ0": 883, "Z0F": 884, "Z15": 885, "Z16": 886, "Z2D": 887, "Z2T": 888, "Z3K": 889, "Z3L": 890, "Z3Q": 891, "Z3U": 892, 
    "Z4K": 893, "Z4R": 894, "Z4S": 895, "Z4U": 896, "Z4V": 897, "Z4W": 898, "Z4Y": 899, "Z57": 900, "Z5J": 901, "Z5L": 902, 
    "Z61": 903, "Z6H": 904, "Z6J": 905, "Z6W": 906, "Z8H": 907, "Z8T": 908, "Z9D": 909, "Z9E": 910, "Z9H": 911, "Z9K": 912, "Z9L": 913, "Z9M": 914, "Z9N": 915, "Z9W": 916, "ZB0": 917, "ZB1": 918, "ZB2": 919, "ZB3": 920, "ZCD": 921, "ZCZ": 922, "ZD0": 923, "ZDC": 924, "ZDO": 925, "ZEE": 926, "ZEL": 927, "ZGE": 928, "ZMR": 929
}


# --- HELPER CLASS: Glycan ---
class Glycan:
    def __init__(self, coor, atom_names, BOND_CUTOFF=1.85):
        self.coor = np.array(coor)
        self.atom_names = atom_names
        self.BOND_CUTOFF = BOND_CUTOFF
        self.adj_mat = []
        self.edges = []
        self.ring_atom = []
        self.ring_atom_plus = []
        self.calc_adjacency()

        ope = []
        for jj in range(len(self.coor)):
            o = self.calc_ring(jj)
            ope.append(o)
            self.calc_adjacency()

        ring = []
        if len(ope) > 0:
            ring.append(ope[0])
            for jj in range(1, len(ope)):
                if isinstance(ope[jj], bool): continue
                skip = False
                for kk in range(len(ring)):
                    if len(ring[kk]) > 0 and len(ope[jj]) > 0 and ring[kk][0] == ope[jj][0]:
                        skip = True
                if skip: continue
                ring.append(ope[jj])

        valid_rings = [r for r in ring if not isinstance(r, bool) and len(r) > 0]
        self.ring_atom = valid_rings
        self.ring_atom_name, self.ring_com = self.get_ring_atom_name()

    def calc_adjacency(self):
        if len(self.coor) == 0:
            self.adj_mat = np.array([])
            self.edges = []
            return
        dm = distance_matrix(self.coor, self.coor)
        adj_mat = dm < self.BOND_CUTOFF
        np.fill_diagonal(adj_mat, 0)
        edge_list = []
        for ii in range(len(adj_mat)):
            edge_list.append([])
            for jj in range(len(adj_mat)):
                if adj_mat[ii, jj]: edge_list[ii].append(jj)
        self.adj_mat = adj_mat
        self.edges = edge_list

    def visit(self, n, edge_list, visited, st):
        if n == st and visited[st] == True: return [n]
        visited[n] = True
        arr = []
        if n >= len(edge_list): return False
        for e in edge_list[n]:
            try: edge_list[e].remove(n)
            except ValueError: continue
            r = self.visit(e, edge_list, visited, st)
            if isinstance(r, list):
                arr.append(n)
                for j in r: arr.append(j)
        if not arr: return False
        return arr

    def calc_ring(self, i):
        ring = self.visit(i, copy.deepcopy(self.edges), np.zeros(len(self.coor), dtype=bool), i)
        ind = 0
        while isinstance(ring, bool):
            ring = self.visit(ind, copy.deepcopy(self.edges), np.zeros(len(self.coor), dtype=bool), ind)
            ind += 1
            if ind >= len(self.coor): break
        if isinstance(ring, list): return np.unique(ring).astype(int)
        return False

    def get_ring_atom_name(self):
        r = []
        com = []
        for jj in self.ring_atom:
            r.append([])
            current_com = np.array([0., 0., 0.])
            for kk in jj:
                r[-1].append(self.atom_names[kk])
                current_com += self.coor[kk]
            if len(jj) > 0: current_com /= len(jj)
            com.append(current_com)
        return r, np.array(com)

# --- HELPER CLASS: LigandGraph ---
class LigandGraph:
    def __init__(self, atoms):
        self.atoms = atoms
        self.coords = np.array([a.coord for a in atoms])
        self.center = np.mean(self.coords, axis=0) if len(self.coords) > 0 else np.array([0,0,0])
        self.size = len(atoms)
        self.atom_names = [a.name for a in atoms]
        self.rdkit_mol = self._to_rdkit()
        self.glycan_obj = Glycan(self.coords, self.atom_names, BOND_CUTOFF=RING_BOND_CUTOFF)

    def _to_rdkit(self):
        mol = Chem.RWMol()
        conf = Chem.Conformer(len(self.atoms))
        for i, atom in enumerate(self.atoms):
            elem = atom.element.capitalize()
            if not elem or len(elem) > 2: elem = "C"
            rd_atom = Chem.Atom(elem)
            mol.AddAtom(rd_atom)
            conf.SetAtomPosition(i, Point3D(float(atom.coord[0]), float(atom.coord[1]), float(atom.coord[2])))
        
        dists = distance_matrix(self.coords, self.coords)
        rows, cols = np.where((dists < BOND_DIST) & (dists > 0))
        bonds = set()
        for r, c in zip(rows, cols):
            if r < c: bonds.add((int(r), int(c)))
        for r, c in bonds: mol.AddBond(r, c, Chem.BondType.SINGLE)
            
        mol_obj = mol.GetMol()
        mol_obj.AddConformer(conf)
        try:
            mol_obj.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol_obj) 
        except: pass
        return mol_obj

# --- CORE FUNCTIONS ---
def load_structure(path, name):
    """
    Loads a structure from PDB or CIF file.
    Robust to case-sensitivity in file extensions.
    """
    # specific check for .cif (case insensitive) to use MMCIFParser
    if path.suffix.lower() == '.cif':
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
        
    try: 
        return parser.get_structure(name, str(path))
    except Exception: 
        return None

def extract_components(structure):
    protein_atoms = [] 
    protein_ca = []
    het_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in STANDARD_AA:
                    for atom in residue:
                        if atom.element.upper() != 'H':
                            protein_atoms.append(atom)
                            if atom.name == 'CA': protein_ca.append(atom)
                else:
                    for atom in residue:
                        if atom.element.upper() != 'H': het_atoms.append(atom)
        break 
    ligands = []
    if het_atoms:
        coords = np.array([a.coord for a in het_atoms])
        dists = distance_matrix(coords, coords)
        G = nx.Graph()
        for i in range(len(het_atoms)): G.add_node(i)
        rows, cols = np.where((dists < BOND_DIST) & (dists > 0))
        for r, c in zip(rows, cols): G.add_edge(r, c)
        for component_indices in nx.connected_components(G):
            comp_atoms = [het_atoms[i] for i in component_indices]
            if len(comp_atoms) >= 3: ligands.append(LigandGraph(comp_atoms))
    return protein_ca, protein_atoms, ligands

# --- METRICS & F_NAT LOGIC ---

def get_residue_id(atom, residue_offset=0):
    """
    Returns a unique tuple identifier for the residue an atom belongs to.
    Applies an integer offset to the residue number.
    """
    # (Chain, ResSeq, ResName)
    p = atom.get_parent()
    # Correctly handle biopython residue id (it's a tuple usually)
    # Standard PDB ID is at index 1
    original_id = p.id[1]
    return (p.get_parent().id, original_id + residue_offset, p.get_resname())

def calculate_residue_offset(ref_ca, pred_ca):
    """
    Calculates the integer offset between Ref and Pred residue numbering.
    Assumes sequences are identical (or highly overlapping).
    Scans offsets from -1000 to +1000 to find max ID overlap.
    Logic mirrors 'fix_num' from utils.py.
    """
    # Extract list of (ResNum, ResName) for both
    # Filter for CA only to be fast
    ref_res = [(a.get_parent().id[1], a.get_parent().get_resname()) for a in ref_ca]
    pred_res = [(a.get_parent().id[1], a.get_parent().get_resname()) for a in pred_ca]
    
    # Create set for O(1) lookup
    ref_set = set(ref_res)
    
    best_offset = 0
    max_overlap = -1
    
    # Try offsets. Usually models start at 1, PDBs at arbitrary numbers.
    # range cover plausible shifts
    for offset in range(-2000, 2000):
        overlap_count = 0
        current_overlap_valid = True
        
        # Check a sample of residues first for speed? No, just check all.
        # Construct shifted pred set
        shifted_matches = 0
        for (r_num, r_name) in pred_res:
            if (r_num + offset, r_name) in ref_set:
                shifted_matches += 1
                
        if shifted_matches > max_overlap:
            max_overlap = shifted_matches
            best_offset = offset
            
        # Optimization: If perfect match found, break
        if max_overlap == len(pred_res):
            break
            
    return best_offset

def calculate_fnat_full(ref_lig, pred_lig, ref_prot_atoms, pred_prot_atoms, pred_offset):
    """
    Calculates F_nat,full: The fraction of native contacts (full ligand to protein).
    A contact is defined as any heavy atom to any heavy atom within 5.0 A. [cite: 811]
    Formula: TP / (TP + FN) -> Recall [cite: 841]
    """
    # 1. Reference Contacts
    # Find all protein residues in Ref that are within 5.0 A of ANY Ref ligand atom
    d_ref = distance_matrix(ref_lig.coords, np.array([a.coord for a in ref_prot_atoms]))
    # min dist per protein atom
    min_d_ref = np.min(d_ref, axis=0) 
    ref_contact_mask = min_d_ref < CONTACT_CUTOFF
    ref_contact_residues = set()
    for i, is_contact in enumerate(ref_contact_mask):
        if is_contact:
            # Ref uses offset=0
            ref_contact_residues.add(get_residue_id(ref_prot_atoms[i], 0))

    # If no native contacts, return 0.0
    if not ref_contact_residues:
        return 0.0

    # 2. Predicted Contacts
    # Find all protein residues in Pred that are within 5.0 A of ANY Pred ligand atom
    d_pred = distance_matrix(pred_lig.coords, np.array([a.coord for a in pred_prot_atoms]))
    min_d_pred = np.min(d_pred, axis=0)
    pred_contact_mask = min_d_pred < CONTACT_CUTOFF
    pred_contact_residues = set()
    for i, is_contact in enumerate(pred_contact_mask):
        if is_contact:
            # Apply Offset to Pred Residues to match Ref Numbering
            pred_contact_residues.add(get_residue_id(pred_prot_atoms[i], pred_offset))

    # 3. Intersection
    tp = len(ref_contact_residues.intersection(pred_contact_residues))
    fn = len(ref_contact_residues) # Denominator is total native contacts (TP + FN)
    
    return tp / fn

def calculate_fnat_res(ref_lig, pred_lig, ref_prot_atoms, pred_prot_atoms, pred_offset):
    """
    Calculates F_nat,res: Fraction of native residue-residue contacts.
    Matches utils.py 'hungarian_fnat':
    1. Identifies contacts for every ring in Ref and Pred.
    2. Builds a generic confusion matrix of overlaps.
    3. Greedily matches rings to MAXIMIZE contact overlap.
    """
    ref_rings = ref_lig.glycan_obj.ring_atom
    pred_rings = pred_lig.glycan_obj.ring_atom

    if not ref_rings or not pred_rings:
        return 0.0

    # --- Step A: Get Contact Lists per Ring ---
    
    # 1. Reference Contacts per Ring
    # List of lists: ref_ring_contacts[i] = [res_id_1, res_id_2...]
    ref_ring_contacts = []
    prot_coords_ref = np.array([a.coord for a in ref_prot_atoms])
    
    for atom_indices in ref_rings:
        ring_coords = ref_lig.coords[atom_indices]
        d_mat = distance_matrix(ring_coords, prot_coords_ref)
        min_d = np.min(d_mat, axis=0)
        contact_indices = np.where(min_d < CONTACT_CUTOFF)[0]
        
        contacts = set()
        for p_idx in contact_indices:
            # Residue ID tuple (Chain, ID, Name)
            contacts.add(get_residue_id(ref_prot_atoms[p_idx], 0))
        ref_ring_contacts.append(list(contacts))

    # 2. Predicted Contacts per Ring
    pred_ring_contacts = []
    prot_coords_pred = np.array([a.coord for a in pred_prot_atoms])
    
    for atom_indices in pred_rings:
        ring_coords = pred_lig.coords[atom_indices]
        d_mat = distance_matrix(ring_coords, prot_coords_pred)
        min_d = np.min(d_mat, axis=0)
        contact_indices = np.where(min_d < CONTACT_CUTOFF)[0]
        
        contacts = set()
        for p_idx in contact_indices:
            # Apply Offset here to align numbering
            contacts.add(get_residue_id(pred_prot_atoms[p_idx], pred_offset))
        pred_ring_contacts.append(list(contacts))

    # --- Step B: Build Overlap Matrix ---
    # Rows = Ref Rings, Cols = Pred Rings
    # Value = Number of shared contacts
    f_matrix = np.zeros((len(ref_ring_contacts), len(pred_ring_contacts)))
    n_matrix = np.zeros((len(ref_ring_contacts), len(pred_ring_contacts))) # Union size (denominator)

    for r, r_contacts in enumerate(ref_ring_contacts):
        for p, p_contacts in enumerate(pred_ring_contacts):
            # Intersection count
            shared = 0
            for c in r_contacts:
                if c in p_contacts:
                    shared += 1
            f_matrix[r, p] = shared
            # Denominator (Number of native contacts for this Ref ring)
            # Note: utils.py seems to use len(curr[aa]) as denominator logic in loop
            n_matrix[r, p] = len(r_contacts)

    # --- Step C: Greedy Maximization (Matches utils.py logic) ---
    rolling_f = 0.0
    rolling_n = 0.0
    
    # Work on copies
    f_mtx = f_matrix.copy()
    n_mtx = n_matrix.copy()
    
    # Loop to pick best matches
    while True:
        # Find max overlap in matrix
        max_val = np.max(f_mtx)
        
        # Stop if no more overlaps or matrix exhausted
        # (utils.py breaks if np.sum(f) < 1, effectively when max is 0)
        if max_val < 1 and np.sum(n_mtx) == 0:
            break
            
        flat_idx = np.argmax(f_mtx)
        r, c = divmod(flat_idx, f_mtx.shape[1])
        
        # Add to totals
        rolling_f += f_mtx[r, c]
        rolling_n += n_mtx[r, c]
        
        # Mask row and col (set to -1 or 0 to ignore)
        f_mtx[r, :] = -1
        f_mtx[:, c] = -1
        n_mtx[r, :] = 0
        n_mtx[:, c] = 0

        # utils.py condition: if np.sum(f) < 1 break
        # We check at start of loop, but this handles the case where we just masked the last one
        if np.max(f_mtx) < 0: 
            break

    # Pick up remaining Ref rings that had 0 matches (Denominator still counts!)
    # utils.py: "while np.sum(n) > 0"
    while np.sum(n_mtx) > 0:
        flat_idx = np.argmax(n_mtx)
        r, c = divmod(flat_idx, n_mtx.shape[1])
        
        rolling_n += n_mtx[r, c]
        
        n_mtx[r, :] = 0
        n_mtx[:, c] = 0

    if rolling_n == 0: 
        return 0.0
        
    return rolling_f / rolling_n

def calculate_rmsd_from_mols(ref_mol, pred_mol):
    """Core RDKit RMSD calculation between two Mol objects."""
    try:
        mcs = rdFMCS.FindMCS([ref_mol, pred_mol], timeout=2, 
                             matchValences=False, ringMatchesRingOnly=True, 
                             completeRingsOnly=True)
        if not mcs.smartsString: return 100.0

        pattern = Chem.MolFromSmarts(mcs.smartsString)
        ref_match = ref_mol.GetSubstructMatch(pattern)
        pred_match = pred_mol.GetSubstructMatch(pattern)
        
        if not ref_match or not pred_match: return 100.0

        atom_map = list(zip(pred_match, ref_match))
        rms = Chem.rdMolAlign.CalcRMS(pred_mol, ref_mol, map=[atom_map])
        return rms
    except Exception:
        return 100.0

def calculate_lrms(ref_lig, pred_lig):
    """Global Ligand RMSD (Position dependent). [cite: 848]"""
    return calculate_rmsd_from_mols(ref_lig.rdkit_mol, pred_lig.rdkit_mol)

def calculate_rirms(ref_lig, pred_lig):
    """
    Ring RMSD.
    Matches paper implementation: Uses Greedy matching on Ring COMs.
    """
    ref_coms = ref_lig.glycan_obj.ring_com
    pred_coms = pred_lig.glycan_obj.ring_com
    
    if len(ref_coms) == 0 or len(pred_coms) == 0: 
        return 10.0

    # 1. Build Cost Matrix (Squared Euclidean Distance)
    cost_mtx = np.zeros((len(ref_coms), len(pred_coms)))
    for r, rc in enumerate(ref_coms):
        for p, pc in enumerate(pred_coms):
            cost_mtx[r, p] = np.sum((rc - pc)**2)

    # 2. Greedy Assignment (Matches utils.py logic)
    sq_diffs = []
    # We work on a copy to mask values with infinity
    mtx = cost_mtx.copy()
    
    # Iterate up to the number of possible pairs
    max_pairs = min(mtx.shape)
    
    for _ in range(max_pairs):
        # Find the global minimum in the matrix
        min_val = np.min(mtx)
        
        # If matrix is all inf (or empty), we are done
        if min_val == np.inf: 
            break
            
        # Get index of minimum
        flat_idx = np.argmin(mtx)
        r, c = divmod(flat_idx, mtx.shape[1])
        
        # Record the squared error
        sq_diffs.append(min_val)
        
        # Mask this row and column (Greedy step)
        mtx[r, :] = np.inf
        mtx[:, c] = np.inf

    if not sq_diffs: 
        return 10.0
        
    return np.sqrt(np.sum(sq_diffs) / len(sq_diffs))

def calculate_dockqc(fnat_res, fnat_full, lrms, rirms):
    """
    DockQC Calculation.
    Formula: 1/3 * ( 1/2*(Fnat_res + Fnat_full) + rRMS_scaled + LRMS_scaled ) [cite: 876]
    """
    fnat_avg = 0.5 * (fnat_res + fnat_full)
    lrms_scaled = 1.0 / (1.0 + (lrms / D_LRMS)**2)
    rrms_scaled = 1.0 / (1.0 + (rirms / D_RRMS)**2)
    return (fnat_avg + rrms_scaled + lrms_scaled) / 3.0

def get_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2: return np.mean(a), 0.0
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

# --- MAIN PROCESSING ---
def process_target(target_folder, pdb_dir, args):
    target_name = Path(target_folder).name
    
    # --- CASE-INSENSITIVE SEARCH for Reference Files ---
    ref_files = []
    pdb_path_obj = Path(pdb_dir)
    
    if pdb_path_obj.exists():
        for f in pdb_path_obj.iterdir():
            if f.is_file():
                if f.name.lower().startswith(target_name.lower()) and \
                   f.suffix.lower() in ['.cif', '.pdb']:
                    ref_files.append(f)
    
    ref_files = sorted(ref_files)
    
    if not ref_files: return []
    
    # Load Reference
    ref_struct = load_structure(ref_files[0], "REF")
    if not ref_struct: return []
    
    ref_ca, ref_prot_atoms, ref_ligands = extract_components(ref_struct)
    if not ref_ligands: return [] 

    # Determine Prediction Files based on format
    if args.format == 'af3':
        pred_files = list(Path(target_folder).glob("seed-1_sample-*/model.cif"))
    elif args.format == 'boltz':
        pred_files = list(Path(target_folder).glob(f"*model_*.cif"))
    elif args.format == 'chai':
        pred_files = list(Path(target_folder).glob("pred.model_*.cif"))
    else:
        pred_files = []
        
    results = []
    
    for pred_path in pred_files:
        model_id = pred_path.stem
        # Load Prediction
        pred_struct = load_structure(pred_path, "PRED")
        if not pred_struct: continue
        
        pred_ca, pred_prot_atoms, pred_ligands = extract_components(pred_struct)
        if not pred_ligands: continue
        
        # --- ALIGNMENT & OFFSET CORRECTION ---
        residue_offset = calculate_residue_offset(ref_ca, pred_ca)
        
        # Identify Pocket Residues in Reference for Alignment
        ref_lig_coords_all = np.vstack([l.coords for l in ref_ligands])
        ref_ca_coords = np.array([a.coord for a in ref_ca])
        
        dists = distance_matrix(ref_lig_coords_all, ref_ca_coords)
        min_dists_to_ligand = np.min(dists, axis=0) 
        pocket_mask = min_dists_to_ligand < POCKET_CUTOFF 
        
        align_indices = np.where(pocket_mask)[0]
        max_idx = min(len(ref_ca), len(pred_ca))
        align_indices = [i for i in align_indices if i < max_idx]
        
        if len(align_indices) < 3:
            align_indices = list(range(max_idx))

        ref_atoms_align = [ref_ca[i] for i in align_indices]
        pred_atoms_align = [pred_ca[i] for i in align_indices]
        
        sup = Superimposer()
        sup.set_atoms(ref_atoms_align, pred_atoms_align)
        sup.apply(pred_struct.get_atoms())

        for pl in pred_ligands:
            pl.coords = np.array([a.coord for a in pl.atoms])
            pl.center = np.mean(pl.coords, axis=0)
            pl.glycan_obj = Glycan(pl.coords, pl.atom_names, BOND_CUTOFF=RING_BOND_CUTOFF)
            pl.rdkit_mol = pl._to_rdkit()
            
        # --- LIGAND MATCHING ---
        cost_mtx = np.full((len(ref_ligands), len(pred_ligands)), np.inf)
        
        for r, ref in enumerate(ref_ligands):
            for p, pred in enumerate(pred_ligands):
                if abs(ref.size - pred.size) <= 1:
                    dist = np.linalg.norm(ref.center - pred.center)
                    cost_mtx[r, p] = dist
                else:
                    cost_mtx[r, p] = np.inf

        matches = []
        curr_mtx = cost_mtx.copy()
        max_possible_matches = min(curr_mtx.shape)
        
        for _ in range(max_possible_matches):
            min_val = np.min(curr_mtx)
            if min_val == np.inf: break
            
            flat_idx = np.argmin(curr_mtx)
            r, p = divmod(flat_idx, curr_mtx.shape[1])
            
            matches.append((r, p))
            curr_mtx[r, :] = np.inf
            curr_mtx[:, p] = np.inf
        
        # --- CALCULATE METRICS ---
        dockqc_scores, lrms_scores = [], []
        ilrms_scores = []
        pocket_lrms_scores = []
        ligand_sizes = []  # <--- Added to track sizes

        for (r_idx, p_idx) in matches:
            ref_lig = ref_ligands[r_idx]
            pred_lig = pred_ligands[p_idx]
            
            
            # --- FILTER: ONLY COUNT SCORES IF REF IS A GLYCAN ---
            is_glycan = False
            ref_resnames = set(atom.get_parent().get_resname() for atom in ref_lig.atoms)
            
            for rname in ref_resnames:
                if rname in MONO_TYPE_MAP:
                    is_glycan = True
                    break
            
            if not is_glycan:
                continue

            
            # A. Calculate Main Metrics
            lrms = calculate_lrms(ref_lig, pred_lig)
            rirms = calculate_rirms(ref_lig, pred_lig)
            
            fnat_full = calculate_fnat_full(ref_lig, pred_lig, ref_prot_atoms, pred_prot_atoms, residue_offset)
            fnat_res = calculate_fnat_res(ref_lig, pred_lig, ref_prot_atoms, pred_prot_atoms, residue_offset)
            
            dqc = calculate_dockqc(fnat_res, fnat_full, lrms, rirms)
            
            internal_lrms = 100.0
            try:
                ref_mol = ref_lig.rdkit_mol
                pred_mol = pred_lig.rdkit_mol
                mcs = rdFMCS.FindMCS([ref_mol, pred_mol], timeout=2, 
                                     matchValences=False, ringMatchesRingOnly=True, 
                                     completeRingsOnly=True)
                if mcs.smartsString:
                    pattern = Chem.MolFromSmarts(mcs.smartsString)
                    ref_match = ref_mol.GetSubstructMatch(pattern)
                    pred_match = pred_mol.GetSubstructMatch(pattern)
                    if ref_match and pred_match:
                        atom_map = list(zip(pred_match, ref_match))
                        internal_lrms = Chem.rdMolAlign.GetBestRMS(pred_mol, ref_mol, map=[atom_map])
            except: pass

            # B. Pocket Metrics
            pocket_lrms = 100.0
            try:
                ref_ca_coords_all = np.array([a.coord for a in ref_ca])
                dists_to_lig = distance_matrix(ref_lig.coords, ref_ca_coords_all)
                min_dists = np.min(dists_to_lig, axis=0)
                local_pocket_mask = min_dists < POCKET_CUTOFF
                
                local_ref_atoms = [ref_ca[i] for i, m in enumerate(local_pocket_mask) if m]
                local_pred_atoms = [pred_ca[i] for i, m in enumerate(local_pocket_mask) if m]
                
                if len(local_ref_atoms) >= 3:
                    sup_pocket = Superimposer()
                    sup_pocket.set_atoms(local_ref_atoms, local_pred_atoms)
                    rot, tran = sup_pocket.rotran
                    
                    pred_coords_transformed = np.dot(pred_lig.coords, rot) + tran
                    
                    pred_mol_pocket = copy.deepcopy(pred_lig.rdkit_mol)
                    conf = pred_mol_pocket.GetConformer()
                    for i, (x,y,z) in enumerate(pred_coords_transformed):
                        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
                        
                    pocket_lrms = calculate_rmsd_from_mols(ref_lig.rdkit_mol, pred_mol_pocket)
            except Exception: pass

            dockqc_scores.append(dqc)
            lrms_scores.append(lrms)
            ilrms_scores.append(internal_lrms)
            pocket_lrms_scores.append(pocket_lrms)
            ligand_sizes.append(ref_lig.size) # <--- Record size

        if not dockqc_scores: continue

        results.append({
            'Target': target_name,
            'Model': model_id,
            'DockQC': dockqc_scores,
            'LRMS': lrms_scores,
            'Internal_RMSD': ilrms_scores,
            'Pocket_LRMS': pocket_lrms_scores,
            'Ligand_Sizes': ligand_sizes # <--- Added to results
        })
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("pdb", type=str)
    parser.add_argument("out", type=str)
    parser.add_argument("--format", type=str, default="af3", choices=["af3", "boltz", "chai"])
    parser.add_argument("--workers", type=int, default=cpu_count())
    args = parser.parse_args()
    
    targets = [f for f in Path(args.data).iterdir() if f.is_dir()]
    print(f"Processing {len(targets)} targets with {args.workers} workers...")
    
    func = partial(process_target, pdb_dir=args.pdb, args=args)
    
    all_rows = []
    with Pool(args.workers) as p:
        with tqdm(total=len(targets)) as pbar:
            for res in p.imap_unordered(func, targets):
                all_rows.extend(res)
                pbar.update()
                
    if not all_rows:
        print("No results found.")
        sys.exit(0)
    
    # --- Write to CSV ---
    print(f"Writing detailed results to {args.out}...")
    with open(args.out, 'w', newline='') as csvfile:
        fieldnames = ['Target', 'Model', 'DockQC', 'LRMS', 'Internal_RMSD', 'Pocket_LRMS']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in all_rows:
            num_ligands = len(row['DockQC'])
            for i in range(num_ligands):
                writer.writerow({
                    'Target': row['Target'],
                    'Model': row['Model'],
                    'DockQC': row['DockQC'][i],
                    'LRMS': row['LRMS'][i],
                    'Internal_RMSD': row['Internal_RMSD'][i],
                    'Pocket_LRMS': row['Pocket_LRMS'][i]
                })

    # --- Summary Report ---
    all_lrms = []
    all_dockqc = []
    all_internal = []
    all_pocket_lrms = []
    all_sizes = [] # <--- Track sizes for global stats

    for row in all_rows:
        all_lrms.extend(row['LRMS'])
        all_dockqc.extend(row['DockQC'])
        all_internal.extend(row['Internal_RMSD'])
        all_pocket_lrms.extend(row['Pocket_LRMS'])
        all_sizes.extend(row['Ligand_Sizes']) 

    all_lrms = np.array(all_lrms)
    all_dockqc = np.array(all_dockqc)
    all_internal = np.array(all_internal)
    all_pocket_lrms = np.array(all_pocket_lrms)
    all_sizes = np.array(all_sizes) 

    def print_ratio(name, arr, threshold, less_than=True):
        if len(arr) == 0:
            print(f"{name:<45} N/A (No samples)")
            return
        if less_than:
            binary = (arr < threshold)
        else:
            binary = (arr >= threshold)
        
        mean, h = get_ci(binary)
        print(f"{name:<45} {mean:.3f} +/- {h:.3f}")

    print("\n--- METRICS REPORT (ALL SUGARS) ---")
    
    print_ratio("Ratio Mean Global LRMS < 2.0 Å:", all_lrms, 2.0)
    print_ratio("DockQC > 0.80:", all_dockqc, 0.80, less_than=False)
    
    if len(all_dockqc) > 0:
        mean_dqc, h_dqc = get_ci(all_dockqc)
        print(f"{'Average DockQC Score:':<45} {mean_dqc:.3f} +/- {h_dqc:.3f}")

    print_ratio("Ratio Mean Pocket LRMS < 2.0 Å:", all_pocket_lrms, 2.0)
    print_ratio("Ratio Mean Internal Ligand RMSD < 1.0 Å:", all_internal, 1.0)
    
if __name__ == "__main__":
    main()
