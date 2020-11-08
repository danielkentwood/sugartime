# sugar
### Time series forecasting and insulin optimization for T1D

Most people never have to think about balancing glucose and insulin in their blood. Diabetes patients have to consciously assume this role because their pancreas isn't doing it for them. It can be exhausting. There is a lot of guesswork that goes into administering the correct amount of insulin at the correct time in order to counter the rise in glucose that will follow a meal.

The **sugar** application is designed to learn how your body responds to food and to insulin. After it has learned this, you can tell it the timing and amount of carbs you plan on eating, and it will offer a suggested insulin bolus amount and timing. This is the model's best guess as to which insulin dosage will keep your blood glucose within the optimal range. 

Here is an example of the app UI:
![Optimization example](./src/images/optim_example.png)

### A few current limitations:
I set out to build something useful for a family member who has diabetes. After lots of experiments, exploration, and tinkering, I've come to an understanding of why such an application doesn't exist yet. It is a really difficult problem. So this app is currently only a proof-of-concept. Here are its primary limitations:
 * The only time-series model available to use is a NARX (nonlinear autoregressive with exogenous regressors) design. Any regression model available within Scikit-Learn can act as the base algorithm. Currently, I've got a vanilla support vector machine hardcoded as the base algorithm, but this can be changed. The NARX implementation relies on the fireTS package (https://github.com/jxx123/fireTS). I plan on adding more model options in the future. For example, RNNs have had some good success with time series forecasting. 
 * The data set is impoverished; it consists of blood glucose readings, carbohydrates consumed, and insulin injected. It's quite possible that my goal (accurate forecasting with a 60-90 minute horizon) isn't reachable without a dataset that is richer in features. Blood glucose is influenced by more than just carbs and insulin, and the effort to train a model without these other predictors may be doomed to hit a low ceiling in terms of model accuracy. There are richer datasets out there with predictors like sleep, heart rate, exercise, etc. I wanted to start with the most stripped down data set in order to maximize the utility to the highest number of people. Most people don't constantly record their heart rate.
 * This app is optimized for dealing with continuous blood glucose monitors (i.e., reading every 5 minutes). More specifically, the app currently only knows how to interpret raw historical data from the Clarity monitor and the Tandem glucose pump. More devices will be added in the future. 
 * There is no way to get real time forecasting since the app only takes historical data. Real time continuous forecasting would be awesome, but most device manufacturers make this difficult by adding a 3-hour lag to the data you can access from their APIs.  
 * The bolus optimization approach is currently a brute force grid search that only allows for one bolus. After adding the possibility for more than one bolus (e.g., staggered bolus for high protein/fat meals), brute force might become a bottleneck; stochastic approaches will be more appropriate. 
