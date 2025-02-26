"""Icestupa class object definition
"""

# External modules
import pickle
pickle.HIGHEST_PROTOCOL = 4  # For python version 2.7
import pandas as pd
import sys, os, math, json
import numpy as np
import logging
import pytz
from tqdm import tqdm
from codetiming import Timer
from datetime import timedelta
from pvlib import atmosphere

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.methods.solar import get_solar
from src.utils.settings import config
from src.plots.data import plot_input

# Module logger
logger = logging.getLogger("__main__")
logger.propagate = False

class Icestupa:
    # def __init__(self, location=None):
    def __init__(self, SITE, FOLDER):

        with open("constants.json") as f:
            CONSTANTS = json.load(f)

        # SITE, FOLDER = config(location)

        initialize = [CONSTANTS, SITE, FOLDER]

        for dictionary in initialize:
            for key in dictionary:
                setattr(self, key, dictionary[key])
                logger.info(f"%s -> %s" % (key, str(dictionary[key])))

        # Initialize input dataset
        self.df = pd.read_csv(self.input + "aws.csv", sep=",", header=0, parse_dates=["time"])
        self.start_date = self.df.time[0]
        self.expiry_date = self.df.time[self.df.shape[0]-1]

        logger.debug(self.df.head())
        logger.debug(self.df.tail())

    # Imported methods
    from src.models.methods._self_attributes import self_attributes
    from src.models.methods._area import get_area
    from src.models.methods._temp import get_temp, test_get_temp
    from src.models.methods._energy import get_energy, test_get_energy
    from src.models.methods._figures import summary_figures

    def read_input(self):  # Use processed input dataset

        self.df = pd.read_hdf(self.input_sim  + "/input.h5", "df")

        if self.df.isnull().values.any():
            logger.warning("\n Null values present\n")

    def read_output(self):  # Reads output

        self.df = pd.read_hdf(self.output + "output.h5", "df")

        self.self_attributes()


    # @Timer(text="Simulation executed in {:.2f} seconds", logger=logging.NOTSET)
    def sim_air(self, test=False):

        """Solar radiation"""
        solar_df = get_solar(
            coords=self.coords,
            start=self.start_date,
            end=self.df["time"].iloc[-1],
            DT=self.DT,
            alt=self.alt,
        )
        self.df = pd.merge(solar_df, self.df, on="time", how="left")
        # self.df["tau_atm"] = self.df["tau_atm"] * self.df["SW_global"]/self.df["ghi"]
        if self.df.isna().values.any():
            logger.warning(self.df[self.df.columns].isna().sum())
            self.df= self.df.interpolate(method='ffill', axis=0)
            logger.warning(f"Filling nan values created by solar module\n")

        self.df.loc[0, "tau_atm"] = 0
        for i in range(1, self.df.shape[0]):
            if (self.df.loc[i, "SW_global"] > 20): #Night time 
                self.df.loc[i, "tau_atm"] = self.df.loc[i,"SW_global"]/self.df.loc[i,"SW_extra"]
                # logger.error(f"Time {self.df.loc[i, "time"]:.0f}, SW_global {self.df.loc[i, "SW_global"]:.0f}, SW_extra {self.df.loc[i,"SW_extra"]:.0f}, transmittivity {self.df.loc[i, "tau_atm"]:.0f}\n")
            else:
                self.df.loc[i, "tau_atm"] = self.df.loc[i-1, "tau_atm"]
            # print(f"Time {self.df.time[i]}, transmittivity {self.df.tau_atm[i]:.2f}\n")


            # Hock, Regine, and Björn Holmgren. “A Distributed Surface Energy-Balance Model for Complex Topography and Its Application to Storglaciären, Sweden.” Journal of Glaciology 51, no. 172 (January 2005): 25–36. https://doi.org/10.3189/172756505781829566.
            if self.df.loc[i, "tau_atm"] >= 0.8:
                self.df.loc[i, "SW_diffuse"] = 0.15 * self.df.loc[i, "SW_global"]
            elif (self.df.loc[i, "tau_atm"]<= 0.15):
                self.df.loc[i, "SW_diffuse"] = self.df.loc[i, "SW_global"]
            else:
                self.df.loc[i, "SW_diffuse"] = 0.929 + 1.134*self.df.loc[i, "tau_atm"] 
                - 5.111 * self.df.loc[i,"tau_atm"]**2 + 3.106 * self.df.loc[i,"tau_atm"]**3
                self.df.loc[i, "SW_diffuse"] *= self.df.loc[i, "SW_global"]

        logger.warning(f"Estimated atmospheric transmittivity {self.df.tau_atm.mean():.2f}\n")
        # logger.error(f"Min TOA {self.df.SW_extra.min():.2f}\n")
        self.df["SW_direct"] = self.df["SW_global"] - self.df["SW_diffuse"]

        """Pressure"""
        self.df["press"] = atmosphere.alt2pres(self.alt) / 100
        logger.warning(f"Estimated pressure from altitude\n")

        self.df = self.df.round(3)

        # Initialisaton for sites
        all_cols = [
            "T_F",
            "T_s",
            "T_bulk",
            "f_cone",
            "ice",
            "iceV",
            "sub",
            "vapour",
            "melted",
            "delta_T_s",
            "wastewater",
            "Qtotal",
            "SW",
            "LW",
            "Qs",
            "Ql",
            "Qf",
            "Qg",
            "meltwater",
            "A_cone",
            "h_cone",
            "r_cone",
            "dr",
            # "snow2ice",
            # "rain2ice",
            "dep",
            "j_cone",
            "wasted",
            "fountain_froze",
            "Qt",
            "Qmelt",
            "Qfreeze",
            "input",
            "event",
            "rho_air",
        ]

        for column in all_cols:
            if column in ["event"]:
                self.df[column] = np.nan
            else:
                self.df[column] = 0

        # Resample to daily minimum temperature
        daily_min_temps = self.df.set_index("time")['temp'].resample('D').min()

        # Find longest consecutive period
        current_period = 0
        start_date_list = []
        crit_temp = 0

        for date, temp in daily_min_temps.items():
            if temp < crit_temp:
                current_period += 1
                if current_period == self.minimum_period:
                    start_date_list.append(date - pd.DateOffset(days=current_period - 1))
            else:
                current_period = 0

        logger.warning(f"Cold windows: {start_date_list}")

        if not start_date_list:
            logger.warning("No cold windows found. Setting iceV_max to -99.")
            results_dict = {
                "iceV_max": 0,
                "survival_days": 0
            }
            print("Summary of results for %s :" %(self.name))
            for var in sorted(results_dict.keys()):
                print("\t%s: %r" % (var, results_dict[var]))

            with open(self.output + "results.json", "w") as f:
                json.dump(results_dict, f, sort_keys=True, indent=4)
            return  # Exit the method early

        self.self_attributes()

        day_index = self.df.index[self.df['time'].dt.strftime('%Y-%m-%d')==start_date_list[0].strftime('%Y-%m-%d')][0]

        # Initialise first model time step
        self.df.loc[day_index, "h_cone"] = self.h_i
        self.df.loc[day_index, "r_cone"] = self.R_F
        self.df.loc[day_index, "SW_diffuse"] = 0
        self.df.loc[day_index, "dr"] = self.DX
        self.df.loc[day_index, "s_cone"] = self.df.loc[day_index, "h_cone"] / self.df.loc[day_index, "r_cone"]
        V_initial = math.pi / 3 * self.R_F ** 2 * self.h_i
        self.df.loc[day_index +1, "rho_air"] = self.RHO_I
        self.df.loc[day_index + 1, "ice"] = V_initial* self.df.loc[day_index + 1, "rho_air"]
        self.df.loc[day_index + 1, "iceV"] = V_initial
        self.df.loc[day_index + 1, "input"] = self.df.loc[day_index + 1, "ice"]

        logger.warning(
            "Initialise: time %s, radius %.3f, height %.3f, iceV %.3f\n"
            % (
                self.df.loc[day_index, "time"],
                self.df.loc[day_index, "r_cone"],
                self.df.loc[day_index, "h_cone"],
                self.df.loc[day_index + 1, "iceV"],
            )
        )

        pbar = tqdm(total = self.df.shape[0])
        pbar.set_description("%s AIR" % self.name)

        i = day_index+1
        pbar.update(i)
        end = self.df.shape[0]-1

        while i <= end:

            ice_melted = self.df.loc[i, "iceV"] < self.V_dome

            if ice_melted:
                # No further cold windows
                if self.df.loc[i, "time"] > start_date_list[-1]:
                    logger.warning("\tNo further cold windows after %s\n" %self.df.loc[i, "time"] )

                    col_list = [
                        "dep",
                        # "snow2ice",
                        "fountain_froze",
                        "wasted",
                        "sub",
                        "melted",
                    ]
                    for column in col_list:
                        self.df.loc[i - 1, column] = 0

                    # last_hour = i - 1
                    # self.df = self.df[1:i]
                    # self.df = self.df.reset_index(drop=True)
                    pbar.update(end - i)

                    # Full Output
                    self.df.to_hdf(
                        self.output  + "/output.h5",
                        key="df",
                        mode="w",
                    )
                    break
                else:
                    for day in start_date_list:
                        if day >= self.df.loc[i+1, "time"]: 
                            day_index = self.df.index[self.df['time'].dt.strftime('%Y-%m-%d')==day.strftime('%Y-%m-%d')][0]
                            pbar.update(day_index - i)
                            i = day_index
                            logger.warning("\tNext cold window at %s\n" % self.df.loc[day_index, "time"])
                            break

                    # Initialise first model time step
                    self.df.loc[i-1, "h_cone"] = self.h_i
                    self.df.loc[i-1, "r_cone"] = self.R_F
                    self.df.loc[i-1, "dr"] = self.DX
                    self.df.loc[i-1, "s_cone"] = self.df.loc[0, "h_cone"] / self.df.loc[0, "r_cone"]
                    V_initial = math.pi / 3 * self.R_F ** 2 * self.h_i
                    self.df.loc[i, "rho_air"] = self.RHO_I
                    self.df.loc[i, "ice"] = V_initial* self.df.loc[i, "rho_air"]
                    self.df.loc[i, "iceV"] = V_initial
                    self.df.loc[i, "input"] = self.df.loc[i, "ice"]

                    logger.warning(
                        "Initialise: time %s, radius %.3f, height %.3f, iceV %.3f\n"
                        % (
                            self.df.loc[i-1, "time"],
                            self.df.loc[i-1, "r_cone"],
                            self.df.loc[i-1, "h_cone"],
                            self.df.loc[i, "iceV"],
                        )
                    )

            self.get_area(i)

            # # Precipitation 
            # if self.df.loc[i, "ppt"] > 0:

            #     if self.df.loc[i, "temp"] < self.T_PPT:
            #         self.df.loc[i, "snow2ice"] = (
            #             self.RHO_W
            #             * self.df.loc[i, "ppt"]
            #             / 1000
            #             * math.pi
            #             * math.pow(self.df.loc[i, "r_cone"], 2)
            #         )
            #     else:
            #     # If rain add to discharge and change temperature
            #         self.df.loc[i, "rain2ice"] = (
            #             self.RHO_W
            #             * self.df.loc[i, "ppt"]
            #             / 1000
            #             * math.pi
            #             * math.pow(self.df.loc[i, "r_cone"], 2)
            #         )
            #         # self.df.loc[i, "Discharge"] += self.df.loc[i, "rain2ice"]/60
            #         self.df.loc[i, "snow2ice"] = 0
            #         logger.info(f"Rain event on {self.df.time.loc[i]} with temp {self.df.temp.loc[i]}")
            # else:
            #     self.df.loc[i, "snow2ice"] = 0

            if test:
                self.test_get_energy(i)
            else:
                self.get_energy(i)

            if test:
                self.test_get_temp(i)
            else:
                self.get_temp(i)

            # Sublimation and deposition
            if self.df.loc[i, "Ql"] < 0:
                L = self.L_S
                self.df.loc[i, "sub"] = -(
                    self.df.loc[i, "Ql"] * self.DT * self.df.loc[i, "A_cone"] / L
                )
            else:
                L = self.L_S
                self.df.loc[i, "dep"] = (
                    self.df.loc[i, "Ql"] * self.DT * self.df.loc[i, "A_cone"] / self.L_S
                )

                

            """ Quantities of all phases """
            self.df.loc[i + 1, "T_s"] = (
                self.df.loc[i, "T_s"] + self.df.loc[i, "delta_T_s"]
            )
            self.df.loc[i + 1, "meltwater"] = (
                self.df.loc[i, "meltwater"] + self.df.loc[i, "melted"]
            )
            self.df.loc[i + 1, "ice"] = (
                self.df.loc[i, "ice"]
                + self.df.loc[i, "fountain_froze"]
                + self.df.loc[i, "dep"]
                # + self.df.loc[i, "snow2ice"]
                - self.df.loc[i, "sub"]
                - self.df.loc[i, "melted"]
            )

            self.df.loc[i + 1, "vapour"] = (
                self.df.loc[i, "vapour"] + self.df.loc[i, "sub"]
            )
            self.df.loc[i + 1, "wastewater"] = (
                self.df.loc[i, "wastewater"] + self.df.loc[i, "wasted"]
            )

            if self.name in ['guttannen21', 'gangles21']:
                self.df.loc[i + 1, "rho_air"] = self.RHO_I
            else:
                self.df.loc[i + 1, "rho_air"] =(
                        # (self.df.loc[1, "ice"] + self.df.loc[:i, "fountain_froze"].sum()+self.df.loc[:i,"dep"].sum()+self.df.loc[:i,"snow2ice"].sum())
                        (self.df.loc[1, "ice"] + self.df.loc[:i, "fountain_froze"].sum()+self.df.loc[:i,"dep"].sum())
                        /(( self.df.loc[1, "ice"] + self.df.loc[:i, "fountain_froze"].sum()+self.df.loc[:i,
                                                                                                        "dep"].sum())/self.RHO_I)
                        # +(self.df.loc[:i, "snow2ice"].sum()/self.RHO_S))
                )

            self.df.loc[i + 1, "iceV"] = self.df.loc[i + 1, "ice"]/self.df.loc[i+1, "rho_air"]

            self.df.loc[i + 1, "input"] = (
                self.df.loc[i, "input"]
                # + self.df.loc[i, "snow2ice"]
                # + self.df.loc[i, "rain2ice"]
                + self.df.loc[i, "dep"]
                + self.df.loc[i, "Discharge"] * self.DT / 60
            )
            self.df.loc[i + 1, "j_cone"] = (
                self.df.loc[i + 1, "iceV"] - self.df.loc[i, "iceV"]
            ) / (self.df.loc[i, "A_cone"])

            # if test and not ice_melted:
                # logger.error(f"time {self.df.time[i]}, iceV {self.df.iceV[i+1]}")
            i = i+1
            pbar.update(1)

        # Processing full output
        self.df.to_hdf(
            self.output  + "/output.h5",
            key="df",
            mode="w",
        )
        results_dict = {}
        results = [
            "iceV_max",
            # "iceV_sum",
            "survival_days",
        ]
        iceV_max = self.df["iceV"].max()
        # iceV_sum = self.df["iceV"].sum()
        survival_days= self.df.iceV.gt(0).sum()/24

        for var in results:
            results_dict[var] = float(round(eval(var), 1))

        print("Summary of results for %s :" %(self.name))
        for var in sorted(results_dict.keys()):
            print("\t%s: %r" % (var, results_dict[var]))

        with open(self.output + "results.json", "w") as f:
            json.dump(results_dict, f, sort_keys=True, indent=4)
        # print(self.df.loc[i, "time"], self.df.loc[i, "iceV"])
