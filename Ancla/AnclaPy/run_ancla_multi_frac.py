import sys
import argparse
from Ancla import HelperFunctions, Visualize, RegressiveModels, TransformationFunctions
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    #parse the arguments
    
    parser = argparse.ArgumentParser(description='AnclaPy: :) ')
    
    # parser.add_argument('--input', type=str, help='Input file path')
    # parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--reg', type=str, default='linear',
                        help='Type of regression to use (linear or lowess)')
    parser.add_argument('--viz', type=bool, default=True,
                        help = "Visualize the results")
    
    args = parser.parse_args()

    # read all 10 fractions 
    frac1 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac1-calib-averaged.csv")
    frac2 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac2-calib-averaged.csv")
    frac3 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac3-calib-averaged.csv")
    frac4 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac4-calib-averaged.csv")
    frac5 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac5-calib-averaged.csv")
    frac6 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac6-calib-averaged.csv")
    frac7 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac7-calib-averaged.csv")
    frac8 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac8-calib-averaged.csv")
    frac9 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac9-calib-averaged.csv")
    frac10 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac10-calib-averaged.csv")

    fractions = []

    for index, fraction in enumerate([frac1, frac2, frac3, frac4, frac5, frac6, frac7, frac8, frac9, frac10]):

        #read the data
        # data = HelperFunctions.read_results(args.input)
        data = fraction
        HelperFunctions.standardScaler(data)
        # data = HelperFunctions.reject_outliers(data)
        
        #fit and transform the data
        if args.reg == 'linear':
            model = RegressiveModels.fit_linear(data)
            data = TransformationFunctions.transform_experimental_linear(data, model)
        elif args.reg == 'lowess':
            model = RegressiveModels.fit_lowess(data)
            data = TransformationFunctions.transform_experimental_lowess(data, model)
        else:
            sys.exit("Invalid regressor")
        
        #save the data
        # data.to_csv(args.output)d
        
        #visualize the data
        if args.viz:
            import os
            # create folder 
            os.makedirs(f"D:\plots/{index}", exist_ok=True)
            Visualize.save_visualize_fractions(data, f"D:\plots/{index}")

            