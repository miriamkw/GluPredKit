# Clarke and Parkes Error Grids
Python Implementation of Clarke and Parkes Error Grids for Blood Glucose Accuracy Analysis

The Error Grid Analysis was developed in order to quantify the risk of using measured or predicted values of blood glucose for glucose management of diabetic patients. Analysis is performed by comparing these estimates to the accurate values (which are assumed to be known). The result of comparing these two points is given as a region between A and E, where A means a medically accurate result, B - medically acceptable, C - unnecessary treatment, D - failure to detect a dangerous condition, E - mistaking serious hypoglycemia for hyperglycemia and vice versa.

This repository includes Python procedures for calculating the detailed error grid zones (including the distinction between the upper and lower regions) for the Clarke Error Grid, as well as for the Parkes Error Grid for both type 1 and type 2 diabetes. The "zone accuracy" function combines the results calculated over multiple points into percentages of each zone.

References:

1. Clarke WL, Cox D, Gonder-Frederick LA, Carter W, Pohl SL. Evaluating clinical accuracy of systems for self-monitoring of blood glucose. Diabetes Care 10:622–628,1987

2. Parkes JL, Slatin SL, Pardo S, Ginsberg BH. A New Consensus Error Grid to Evaluate the Clinical Significance of Inaccuracies in the Measurement of Blood Glucose. Diabetes Care 23, no. 8 (August 2000): 1143-48

3. Pfutzner A, Klonoff DC, Pardo S, Parkes JL. Technical Aspects of the Parkes Error Grid. Journal of Diabetes Science and Technology 7, no. 5 (September 2013): 1275-81
