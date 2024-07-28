# Machine Learning Model for Classifying Physical Volunteering Ability

<p align="justify">
This project aims to enhance the selection process for physical volunteering by using machine learning algorithms to classify volunteers based on their ability to perform physically demanding tasks. By analyzing various health-related parameters such as age, chest pain type, blood pressure, cholesterol levels, maximum heart rate, and more, the model determines whether an individual is suitable for roles such as heavy lifting, water distribution, and evacuation assistance during disaster relief operations.
</p>

## Key Features & Technologies

### 1- Data Analysis and Preprocessing: 

  - <p align="justify"> Handling missing values, encoding categorical data, and binarization of features to prepare the dataset. </p>

### 2- Multiple Machine Learning Algorithms:

  - <p align="justify"> Implementation of various machine learning algorithms, including Decision Trees, K-Nearest Neighbors (KNN), Naive Bayes, and Support Vector Machines (SVM), to classify volunteers. </p>

### 3- Comparative Analysis:

  - <p align="justify"> A heavy emphasis on trying out different algorithms and evaluating their performance to understand which algorithm works best for this specific application. The project involved detailed comparison and analysis to identify the most effective model. </p>

### 4- Feature Selection:

  - <p align="justify"> Use of k-Best feature selection and chi-squared tests to enhance model performance by selecting the most relevant features. </p>

### 5- Model Fine-Tuning:

  - <p align="justify"> Optimization of hyperparameters to prevent overfitting and ensure robust model performance. </p>

### 6- Evaluation Metrics: 

  - <p align="justify"> Comprehensive evaluation using metrics such as accuracy, F1-score, precision, recall, AUC, and ROC curves to determine the best-performing model. </p>

### 7- Leveraging Advanced Tools & Libraries:

  - **Python:**

    - The primary language for data analysis, model training, and evaluation.

  - **Jupyter Notebook:** 

    - <p align="justify"> An interactive environment for live code, equations, visualizations, and narrative text. </p>

  - **Pandas:** 
  
    - Data manipulation and analysis library for structured data.

  - **NumPy:** 
    
    - Numerical computing library supporting large, multi-dimensional arrays and matrices.

  - **Scikit-learn:** 
  
    - <p align="justify"> Comprehensive machine learning library for data mining, model selection, preprocessing, and various algorithms. </p>

  - **Matplotlib:** 
  
    - Plotting library for static, animated, and interactive visualizations.

## Machine Learning Model Evaluation

### Overview 

<p align="justify">
The evaluation of the machine learning model was conducted using a comprehensive set of metrics to ensure robust performance. The primary evaluation metrics included Accuracy, F1 Score, Precision, Recall (Sensitivity), AUC (Area Under the Curve), ROC Curve (Receiver Operating Characteristic Curve), and Confusion Matrix. Each model was fine-tuned using specific hyperparameters to optimize performance. For the Decision Tree classifier, min_samples_split, max_depth, min_samples_leaf, and ccp_alpha were adjusted. The K-Nearest Neighbors (KNN) algorithm was tuned using the n_neighbors parameter. The Gaussian Naive Bayes classifier was optimized by adjusting the var_smoothing parameter, while the Support Vector Machine (SVM) was fine-tuned using the C and gamma parameters. Among all the algorithms, the Decision Tree classifier emerged as the best, achieving nearly 100% accuracy in classifying the physical volunteering ability of individuals.
</p>
 
*Note: To see all results, running the project would be required. The images provided here represent a subset of the results obtained during the project.*

### Decision Tree Classifier F1 Score

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1jxYp9et2AB29A7csVm9njEFhJnOCbLp0" alt="Decision Tree Classifier F1 Score" />
</div>

### Decision Tree Classifier Best Max Depth

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1cOEWUuCM7h3LTOzxBkgkLupyz5yF47rC" alt="Decision Tree Classifier Best Max Depth" />
</div>

### K-Nearest Neighbors ROC

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1EsaAmSxZzFbyIXYHpCjcNpzzxLRQGvMv" alt="KNN ROC" />
</div>

### K-Nearest Neighbors Confusion Matrix

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1KAzE3SN5JzabBDCvxKNSNRc08ZokR1cf" alt="KNN Confusion Matrix" />
</div>

### Gaussian Naive Bayes Precision

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=13OJyhmGGo2o_lAjU4b5K5N38kdqFeu8-" alt="GNB Precision" />
</div>

### Gaussian Naive Bayes Recall

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1yYIJEXUUVrfe0QYZJTAzFKmO8T3hBLBI" alt="GNB Recall" />
</div>

### Support Vector Machine Accuracy

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1mwMsz30p1nF6oomc34gjBbC7liBw9dLT" alt="SVM Accuracy" />
</div>

### Support Vector Machine ROC

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=11ztALhEaPyzs83-iOlq7yBQmNh1ZRhEA" alt="SVM ROC" />
</div>

## Contact for Support

For any queries or support related to this project, feel free to contact me at ibrahimsaffarini2001@gmail.com.
