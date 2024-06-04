import sys
import argparse
from Ancla import HelperFunctions, Visualize, RegressiveModels, TransformationFunctions
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #parse the arguments
    
    parser = argparse.ArgumentParser(description='AnclaPy: :) ')
    
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--reg', type=str, default='linear',
                        help='Type of regression to use (linear or lowess)')
    parser.add_argument('--viz', type=bool, default=True,
                        help = "Visualize the results")
    
    args = parser.parse_args()

    #read the data
    data = HelperFunctions.read_results(args.input)
    HelperFunctions.standardScaler(data)
    data = HelperFunctions.reject_outliers(data)
    
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
    data.to_csv(args.output)
    
    #visualize the data
    if args.viz:
        Visualize.visualize_all(data)