# Fuzzy-Systems

### Description
This collection of scripts was created for the scope of an assignment in a Fuzzy Systems cource of the Aristotle University of Thessaloniki during the 2018-19 academic year. The main purpose of these projects is to demonstrate the implementation of a series of different Fuzzy Systems and compare their efficiency to the corresponding Crisp ones. The systems mentioned conern fields of Control Theory, Neural Networks (Regression and Classification Problems).

### Implementations

#### Part 1 - Control Linear System
Construction of a typical PI controller and a fuzzy PI controller to control a given linear system. Creation of usefull Graphs to show functionality and stability of the controlled system.
 
 Block Diagram             |  Fuzzy PI Controller
:-------------------------:|:-------------------------:
![](https://github.com/kosletr/Fuzzy-Systems/blob/master/asafi-ergasia1/pics/fuzzyClosedLoop.jpg) |  ![](https://github.com/kosletr/Fuzzy-Systems/blob/master/asafi-ergasia1/pics/fuzzyPIController.jpg)
  
Efficiency comparison between a typical PI controller vs a fuzzy PI controller. Efficiency can be measured by different means such as smaller rise time and settling time, specifications satisfaction etc.
  
 Typical PI Controller Response       |  Fuzzy PI Controller Response
:-------------------------:|:-------------------------:
![](https://github.com/kosletr/Fuzzy-Systems/blob/master/asafi-ergasia1/pics/stepRespFuzzy_Good.jpg) |  ![](https://github.com/kosletr/Fuzzy-Systems/blob/master/asafi-ergasia1/pics/stepRespLinear.jpg)
  
#### Part 2 - Car's Fuzzy Controller
Creation of a Fuzzy Control System to produce appropriate trajectory from different starting angles. The purpose of this implementation is to appropriately define a set of fuzzy rules to avoid a set of obstacles while remaining at the same time not too far from them.
 
 Starting Angle: 0 degrees | Starting Angle: -45 degrees
:-------------------------:|:-------------------------:
![](https://github.com/kosletr/Fuzzy-Systems/blob/master/asafi-ergasia2/pics/angle0impr.jpg) |  ![](https://github.com/kosletr/Fuzzy-Systems/blob/master/asafi-ergasia2/pics/angle-45impr.jpg)

#### Part 3 - Fuzzy Neural Networks - Regression Problem
Dealing with two Regression Problems using Fuzzy Neural Network techniques such as Fuzzy Clustering Method (FCM), Substractive Clustering Method, as well as many common machine learning tools-techniques such as Cross Validation, Grid Search, Dimensionality Reduction (Relief's Algorithm - Keep Most Significant Features), use of commonly used metrics such as MSE, RMSE, R^2 coefficient and more. 

First Problem - Small Dataset (4 TSK Models)

 Learning Curve Example    | Trained Membership Functions Example
:-------------------------:|:-------------------------:
![](https://github.com/kosletr/Fuzzy-Systems/blob/master/asafi-ergasia3/pics/CCPP/TSK_1_Learning_Curve.jpg) |  ![](https://github.com/kosletr/Fuzzy-Systems/blob/master/asafi-ergasia3/pics/CCPP/TSK_1_MF_after_Training.jpg)

Second Problem - Big Dataset (Grid Search for different number of the most significant Features and number of Fuzzy Rules).

 3D Grid Search MSE Results| 2D Grid Search MSE Results 
:-------------------------:|:-------------------------:
![](https://github.com/kosletr/Fuzzy-Systems/blob/master/asafi-ergasia3/pics/Grid%20Search%20Superconduct/3Dplot_Mean_Error.jpg) |  ![](https://github.com/kosletr/Fuzzy-Systems/blob/master/asafi-ergasia3/pics/Grid%20Search%20Superconduct/Subplots_Mean_Errors.jpg)

After the Grid Search has finished, the optimal model (lowest MSE) is trained.

#### Part 4 - Fuzzy Neural Networks - Classification Problem
Dealing with two Classification Problems using Fuzzy Neural Network techniques such as Fuzzy Clustering Method (FCM), Substractive Clustering Method, as well as many common machine learning tools-techniques such as Cross Validation, Dimensionality Reduction (Relief's Algorithm), use of commonly used metrics such as MSE, RMSE, R^2 coefficient and more.

 First Problem - Small Dataset (4 TSK Models)

 Learning Curve Example    | Confusion Matrix Example
:-------------------------:|:-------------------------:
![](https://github.com/kosletr/Fuzzy-Systems/blob/master/asafi-ergasia4/Plots/Avila/learning_curve_1.jpg) |  ![](https://github.com/kosletr/Fuzzy-Systems/blob/master/asafi-ergasia4/Plots/Avila/Confusion_Matrix5.jpg)

Second Problem - Big Dataset (Grid Search for different number of the most significant Features and number of Fuzzy Rules).

 3D Grid Search MSE Results| 2D Grid Search MSE Results 
:-------------------------:|:-------------------------:
![](https://github.com/kosletr/Fuzzy-Systems/blob/master/asafi-ergasia4/Plots/Isolet%20Grid%20Search/3Dplot_Mean_Error_2.jpg) |  ![](https://github.com/kosletr/Fuzzy-Systems/blob/master/asafi-ergasia4/Plots/Isolet%20Grid%20Search/Subplots_Mean_Errors.jpg)

After the Grid Search has finished, the optimal model (lowest MSE) is trained.

### Setup
The provided code was created using MATLAB R2019a, however older software versions should work fine. All of the .m scripts provided, are  commented for higher readability and maintenance.
