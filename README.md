# Masters-project
Deep learning surrogate models for Askaryan signal generation and subsequent in-ice ray tracing.

## Ray tracing, Classification
- To generate the training data run raytracing_data.py specifying the locations for the saved data files
- To convert the generated data from cartesian coordinates to spherical coordinates use raytracing_sphere_data.ipynb
- To run a random grid search for hyperparameters of the classifier run classifier_train.py
- To train the best model found in the search further, test it and plot the results use classifier_retrain.ipynb
### Performance of the classifier
![image](https://user-images.githubusercontent.com/61893305/172429414-788e85a3-5160-486e-9a9c-168c70c80925.png)


## Ray tracing, Regression
- Data for the regression is generated together with the classification data
- To train the regression model use raytrace_train_single.py with a model defined in raytrace_model_def.py
- To test the trained model and plot results use raytrace_test.ipynb
### Performance of the regression in the case of the first solution for travel time
![sol_0_simpler](https://user-images.githubusercontent.com/61893305/172429690-5fff1f19-f38e-4a00-ae15-e5d52cfacc38.png)

## Signal generation
- To generate the training data run generate_signal_data.py
- To perform a random grid search of hyperparameters use signal_GAN_train.py. Which uses models defined in CWGANGP_model_def.py
- To test the models from the search use signal_GAN_test.py
- To plot results use GAN_results.ipynb
- To test signal generation and scaling use test_signal_gen.ipynb and test_angle_normalization.ipynb

### Example of a generated signal with energy E=1e19eV and 5 degrees away from the Cherenkov cone
![image](https://user-images.githubusercontent.com/61893305/172427497-72950582-00a3-47e1-bb35-c20cb41503d2.png)
### Example of a generated signal with energy E=1e19eV and right on the Cherenkov cone
![image](https://user-images.githubusercontent.com/61893305/172428255-ccdaacb5-ff0d-44f7-a283-dec25cebc3ac.png)