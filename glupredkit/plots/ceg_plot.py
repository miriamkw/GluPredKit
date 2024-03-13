import matplotlib.pyplot as plt
import itertools
import os
import sys
from datetime import datetime
from .base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager

# https://github.com/suetAndTie/ClarkeErrorGrid/blob/master/ClarkeErrorGrid.py
'''
CLARKE ERROR GRID ANALYSIS      ClarkeErrorGrid.py

Need Matplotlib Pyplot


The Clarke Error Grid shows the differences between a blood glucose predictive measurement and a reference measurement,
and it shows the clinical significance of the differences between these values.
The x-axis corresponds to the reference value and the y-axis corresponds to the prediction.
The diagonal line shows the prediction value is the exact same as the reference value.
This grid is split into five zones. Zone A is defined as clinical accuracy while
zones C, D, and E are considered clinical error.

Zone A: Clinically Accurate
    This zone holds the values that differ from the reference values no more than 20 percent
    or the values in the hypoglycemic range (<70 mg/dl).
    According to the literature, values in zone A are considered clinically accurate.
    These values would lead to clinically correct treatment decisions.

Zone B: Clinically Acceptable
    This zone holds values that differe more than 20 percent but would lead to
    benign or no treatment based on assumptions.

Zone C: Overcorrecting
    This zone leads to overcorrecting acceptable BG levels.

Zone D: Failure to Detect
    This zone leads to failure to detect and treat errors in BG levels.
    The actual BG levels are outside of the acceptable levels while the predictions
    lie within the acceptable range

Zone E: Erroneous treatment
    This zone leads to erroneous treatment because prediction values are opposite to
    actual BG levels, and treatment would be opposite to what is recommended.


SYNTAX:
        plot, zone = clarke_error_grid(ref_values, pred_values, title_string)

INPUT:
        ref_values          List of n reference values.
        pred_values         List of n prediciton values.
        title_string        String of the title.

OUTPUT:
        plot                The Clarke Error Grid Plot returned by the function.
                            Use this with plot.show()
        zone                List of values in each zone.
                            0=A, 1=B, 2=C, 3=D, 4=E

EXAMPLE:
        plot, zone = clarke_error_grid(ref_values, pred_values, "00897741 Linear Regression")
        plot.show()

References:
[1]     Clarke, WL. (2005). "The Original Clarke Error Grid Analysis (EGA)."
        Diabetes Technology and Therapeutics 7(5), pp. 776-779.
[2]     Maran, A. et al. (2002). "Continuous Subcutaneous Glucose Monitoring in Diabetic
        Patients" Diabetes Care, 25(2).
[3]     Kovatchev, B.P. et al. (2004). "Evaluating the Accuracy of Continuous Glucose-
        Monitoring Sensors" Diabetes Care, 27(8).
[4]     Guevara, E. and Gonzalez, F. J. (2008). Prediction of Glucose Concentration by
        Impedance Phase Measurements, in MEDICAL PHYSICS: Tenth Mexican
        Symposium on Medical Physics, Mexico City, Mexico, vol. 1032, pp.
        259261.
[5]     Guevara, E. and Gonzalez, F. J. (2010). Joint optical-electrical technique for
        noninvasive glucose monitoring, REVISTA MEXICANA DE FISICA, vol. 56,
        no. 5, pp. 430434.


Made by:
Trevor Tsue
7/18/17

Based on the Matlab Clarke Error Grid Analysis File Version 1.2 by:
Edgar Guevara Codina
codina@REMOVETHIScactus.iico.uaslp.mx
March 29 2013
'''

class Plot(BasePlot):
    def __init__(self):
        super().__init__()

    def __call__(self, models_data):
        if unit_config_manager.use_mgdl:
            unit = "mg/dL"
            max_val = 400
        else:
            unit = "mmol/L"
            max_val = unit_config_manager.convert_value(400)

        plt.figure(figsize=(10, 8))

        for model_data in models_data:
            if unit_config_manager.use_mgdl:
                y_pred = model_data.get('y_pred')
                y_true = model_data.get('y_true')
            else:
                y_pred = [unit_config_manager.convert_value(val) for val in model_data.get('y_pred')]
                y_true = [unit_config_manager.convert_value(val) for val in model_data.get('y_true')]

        ref_values = y_true
        pred_values = y_pred
        title_string = sys.argv[3].split("__")[0].upper()
        #Checking to see if the lengths of the reference and prediction arrays are the same
        assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(len(ref_values), len(pred_values))

        #Checks to see if the values are within the normal physiological range, otherwise it gives a warning
        if max(ref_values) > 400 or max(pred_values) > 400:
            print("Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(max(ref_values), max(pred_values)))
        if min(ref_values) < 0 or min(pred_values) < 0:
            print("Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(min(ref_values),  min(pred_values)))

        #Clear plot
        plt.clf()

        #Set up plot
        plt.scatter(ref_values, pred_values, marker='o', color='red', s=8, alpha=0.5)
        plt.title(title_string + " Clarke Error Grid")
        plt.xlabel("Reference Concentration (mg/dl)")
        plt.ylabel("Prediction Concentration (mg/dl)")
        plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
        plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
        plt.gca().set_facecolor('white')

        #Set axes lengths
        plt.gca().set_xlim([0, 400])
        plt.gca().set_ylim([0, 400])
        plt.gca().set_aspect((400)/(400))

        #Plot zone lines
        plt.plot([0,400], [0,400], ':', c='black')                      #Theoretical 45 regression line
        plt.plot([0, 175/3], [70, 70], '-', c='black')
        #plt.plot([175/3, 320], [70, 400], '-', c='black')
        plt.plot([175/3, 400/1.2], [70, 400], '-', c='black')           #Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
        plt.plot([70, 70], [84, 400],'-', c='black')
        plt.plot([0, 70], [180, 180], '-', c='black')
        plt.plot([70, 290],[180, 400],'-', c='black')
        # plt.plot([70, 70], [0, 175/3], '-', c='black')
        plt.plot([70, 70], [0, 56], '-', c='black')                     #Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
        # plt.plot([70, 400],[175/3, 320],'-', c='black')
        plt.plot([70, 400], [56, 320],'-', c='black')
        plt.plot([180, 180], [0, 70], '-', c='black')
        plt.plot([180, 400], [70, 70], '-', c='black')
        plt.plot([240, 240], [70, 180],'-', c='black')
        plt.plot([240, 400], [180, 180], '-', c='black')
        plt.plot([130, 180], [0, 70], '-', c='black')

        #Add zone titles
        plt.text(30, 15, "A", fontsize=15)
        plt.text(370, 260, "B", fontsize=15)
        plt.text(280, 370, "B", fontsize=15)
        plt.text(160, 370, "C", fontsize=15)
        plt.text(160, 15, "C", fontsize=15)
        plt.text(30, 140, "D", fontsize=15)
        plt.text(370, 120, "D", fontsize=15)
        plt.text(30, 370, "E", fontsize=15)
        plt.text(370, 15, "E", fontsize=15)

        #Statistics from the data
        # zone = [0] * 5
        # for i in range(len(ref_values)):
        #     if (ref_values[i] <= 70 and pred_values[i] <= 70) or (pred_values[i] <= 1.2*ref_values[i] and pred_values[i] >= 0.8*ref_values[i]):
        #         zone[0] += 1    #Zone A

        #     elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
        #         zone[4] += 1    #Zone E

        #     elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or ((ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7/5)*ref_values[i] - 182)):
        #         zone[2] += 1    #Zone C
        #     elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (ref_values[i] <= 175/3 and pred_values[i] <= 180 and pred_values[i] >= 70) or ((ref_values[i] >= 175/3 and ref_values[i] <= 70) and pred_values[i] >= (6/5)*ref_values[i]):
        #         zone[3] += 1    #Zone D
        #     else:
        #         zone[1] += 1    #Zone B

        # return plt, zone
        # print("zone: ", zone)
        file_path = "data/figures/"
        os.makedirs(file_path, exist_ok=True)
        config_str = sys.argv[3].replace(".pkl", "")
        print(sys.argv[3])
        file_name = f'Clarke_Error_Grid_{config_str}.png'
        plt.savefig(file_path + file_name)
        plt.show()

