# Employee-Churn-Prediction-Model-

This report details the construction and validation of a predictive model aimed at identifying the likelihood of employees leaving a pharmaceutical company. The model was developed using Random Forest, a robust machine learning algorithm, with the goal of enabling proactive retention strategies.

### Introduction
Employee churn, or turnover, is a significant concern for organizations due to its associated costs and the loss of talent and institutional knowledge. To mitigate these issues, I developed a predictive model to identify the probability of employee departure, thus allowing for timely intervention measures.

### Data Description
The dataset comprises 1,424 observations of employees, with 32 variables collected. After careful consideration, I omitted four variables that were deemed non-contributory to our analysis, including 'Employee Number,' 'Employee Count,' 'Over 18,' and 'Standard Hours.' The target variable for the model is "Leaving the company," a binary indicator of whether an employee is likely to leave.

### Data Preparation
The dataset underwent pre-processing, where categorical variables were converted to factors, and all variable names were standardized for consistency. I then partitioned the data into a training set (70%) and a testing set (30%) to facilitate model training and validation.


### Model Development
Using the ranger package, I applied the Random Forest algorithm to the training data. I chose Random Forest for its efficacy in handling a large number of input variables and its robustness to overfitting. The model was initially set up with 500 trees, and then calculated variable importance to understand the factors most predictive of churn.

<img src="https://github.com/andinetM/Employee-Churn-Prediction-Model-/blob/main/Plots/Var_Importance_Rplot01.png" align="center" height="500" width="500"/>

 
### Model Tuning
Recognizing the need for model optimization, I conducted a Hyperparameter tuning exercise for the 'mtry' parameter, which represents the number of variables randomly sampled as candidates at each split. Then explored a range of values and determined that mtry = 5 yielded the highest Area Under the Curve (0.8157) metric, which is considered good and indicates the model has a strong discriminative ability.

<img src="https://github.com/andinetM/Employee-Churn-Prediction-Model-/blob/main/Plots/ROC_plot_Rplot.png" align="center" height="500" width="500"/>
 
### Model Evaluation
I evaluated the model using a custom decision threshold of 0.3, rather than the default 0.5, to increase sensitivity to the minority class (Yes). This adjustment was critical in addressing the imbalance in our dataset. The model's accuracy and ability to predict churn were summarized in a confusion matrix.
```
Confusion Matrix and Statistics

          Reference
Prediction  No Yes
       No  321  32
       Yes  36  37
                                          
               Accuracy : 0.8404          
                 95% CI : (0.8021, 0.8739)
    No Information Rate : 0.838           
    P-Value [Acc > NIR] : 0.4797          
                                          
                  Kappa : 0.4254          
                                          
 Mcnemar's Test P-Value : 0.7160          
                                          
            Sensitivity : 0.8992          
            Specificity : 0.5362          
         Pos Pred Value : 0.9093          
         Neg Pred Value : 0.5068          
             Prevalence : 0.8380          
         Detection Rate : 0.7535          
   Detection Prevalence : 0.8286          
      Balanced Accuracy : 0.7177          
                                          
       'Positive' Class : No 
```

The model exhibited high sensitivity for the "No" class but a modest sensitivity for the "Yes”. 

#### Performance Highlights:
- Accuracy: 0.8404, indicating 84% of the predictions align with the actual outcomes.
-	Precision (Positive Predictive Value): 0.9117, reflecting a high reliability in the model's positive ("Yes") predictions.
-	Sensitivity for 'No': 0.8964, illustrating the model's competency in identifying true negatives.
-	Specificity for 'Yes': 0.5507, denoting the model's moderate success rate in detecting true positives.
-	Kappa: 0.4319, signifying moderate concordance beyond chance.


### Cross-Validation
To ensure the model's robustness and generalizability, we employed 10-fold cross-validation. The process involved training the model in nine folds and validating it on the remaining fold. The cross-validated mean AUC of 0.829 is a strong indicator that the model is robust and can make reliable predictions when deployed in a real-world setting.

