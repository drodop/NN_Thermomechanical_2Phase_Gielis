========================================================================

Neural Network models for thermal and mechanical effective property calculation of double phase materials.
The inner architecture is described by the Gielis formula (https://doi.org/10.1007/s00022-015-0269-z)

========================================================================

The directories thermal_R3_predictions and mechanical_R3_predictions contain the trained neural network models, as well as the main files that load the models and run them in order to predict thermal and elastic effective properties.


-> Effective thermal conductivity calculation

        _______________________________

	Directory: thermal_R3_prediction
	Main file: predict_thermal.py
        _______________________________

	Model:  N_T: x_T |--> k
		
		x_T = (Vf,n,rk)

	Input features: 
		Vf  : Volume fraction 
		n   : topology shape
		rk  : material ratio (k1/k2)

	Output:
		k: Effective thermal conductivity


-> Effective elastic moduli calculation (Young and Shear)

        _______________________________

	Directory: mechanical_R3_predictions
	Main file: predict_elastic.py
        _______________________________

	Model:  N_E: x_E |--> (E,G)

		x_E = (Vf,n,rE)

	Input features: 
		Vf  : Volume fraction 
		n   : topology shape
		rE  : material ratio (E1/E2)

	Output:
		(E,G): Effective Young's and Shear moduli


