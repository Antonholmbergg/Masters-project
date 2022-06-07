# Masters-project
Deep learning surrogate models for Askaryan signal generation and subsequent in-ice ray tracing.

## Ray tracing
-To generate the training data run raytracing_data.py specifying the locations for the saved data files
-To convert the generated data from cartesian coordinates to spherical coordinates use raytracing_sphere_data.ipynb
-To run a random grid search for hyperparameters of the classifier run classifier_train.py
-To train the best model found in the search further, test it and plot the results use classifier_retrain.ipynb
-To train the regression model use raytrace_train_single.py with a model defined in raytrace_model_def.py
-To test the trained model and plot results use raytrace_test.ipynb

## Signal generation
-To generate the training data run generate_signal_data.py
-To perform a random grid search of hyperparameters use signal_GAN_train.py. Which uses models defined in CWGANGP_model_def.py
-To test the models from the search use signal_GAN_test.py
-To plot results use GAN_results.ipynb
-To test signal generation and scaling use test_signal_gen.ipynb and test_angle_normalization.ipynb

###Example of a generated signal with energy E=1e19eV and 5 degrees away from the Cherenkov cone
![image](https://user-images.githubusercontent.com/61893305/172427497-72950582-00a3-47e1-bb35-c20cb41503d2.png)
