# sugarTS
 ### Time series forecasting and optimizing insulin for T1D

 It is easy to take a functioning pancreas for granted; most people never have to think about balancing glucose and insulin in their blood. Diabetes (type 1) patients have to consciously assume this role because their pancreas isn't doing it for them. It can be exhausting. There is a lot of guesswork that goes into administering the correct amount of insulin at the correct time in order to counter the rise in glucose that will follow a meal.

 The sugarTS application is designed to learn how your body responds to food and to insulin. After it has learned this, you can tell it when and how many carbs you plan on eating, and it will offer a suggested insulin bolus amount and timing. This is the model's best guess as to which insulin dosage will keep your blood glucose within optimal range. 

 Current limitations:
 * The only model available to use is a NARX (nonlinear autoregressive with exogenous regressors) design; however, any regression model available within SKLearn can act as the base algorithm. Currently, I've got simple linear regression hardcoded as the base algorithm, but this can be changed. The NARX model is implemented in the fireTS package (https://github.com/jxx123/fireTS). I plan on adding more model options in the future. For example, RNNs have had some good success with time series forecasting. 
 * This app is optimized for dealing with continuous blood glucose monitors (i.e., reading every 5 minutes). More specifically, the app currently only knows how to interpret historical data from the Clarity monitor and the Tandem glucose pump. More devices will be added in the future. 
 * This is currently only a proof of principle; there is no way to get real time forecasting since the app only takes historical data. Real time continuous forecasting would be awesome. 
 * The bolus optimization approach is currently a brute force grid search that only allows for one bolus. After adding the possibility for more than one bolus (e.g., staggered bolus for high protein/fat meals), brute force might become a bottleneck; monte carlo or gradient descent approaches will be more appropriate. 
