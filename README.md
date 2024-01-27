# Home Credit Default Risk
The objective of this project is to use historical loan application data to predict whether or not an applicant will be able to repay a loan. This is a standard supervised calssification task:

Supervised: The labels are included in the training data and the goal is to train a model to learn to predict the labels from the features

Classification: The label is a bianary variable, 0(will repay loan on time), 1(will have difficulty repaying loan)
0 - repay O, 1 - repay x

# DATA
These are 7 different sources of data:

1. application_train/application_test: the main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature SK_ID_CURR. The training application data comes with the TARGET indicating 0: the loan was repaid or 1: the loan was not repaid. 

2. bureau: data concerning client's previous credits from other financial institutions. Each previous credit has its own row in bureau, but one loan in the application data can have multiple previous credits.

3. bureau_balance: monthly data about the previous credits in bureau. Each row is one month of a previous credit, and a single previous credit can have multiple rows, one for each month of the credit length.

4. previous_application: previous applications for loans at Home Credit of clients who have loans in the application data. Each current loan in the application data can have multiple previous loans. Each previous application has one row and is identified by the feature SK_ID_PREV.

5. POS_CASH_BALANCE: monthly data about previous point of sale or cash loans clients have had with Home Credit. Each row is one month of a previous point of sale or cash loan, and a single previous loan can have many rows.

6. credit_card_balance: monthly data about previous credit cards clients have had with Home Credit. Each row is one month of a credit card balance, and a single credit card can have many rows.

7. installments_payment: payment history for previous loans at Home Credit. There is one row for every made payment and one row for every missed payment.

# RELATED INFORMATION 
# Exploratory Data Analysis(EDA)¶
EDA is an open-ended process where we calculate statistics and make figures to find trends, anomalies, patterns, or relationships within the data. The goal of EDA is to learn what our data can tell us. It generally starts out with a high level overview, then narrows in to specific areas as we find intriguing areas of the data. The findings may be interesting in their own right, or thy can be used to inform our modeling choices, such as by helping us decide which features to use.

# Encoding Categorical Variables
Before we go any further, we need to deal with pesky categorical variables. A machine learning model unfortunately cannot deal with categorical variables (except for some models such as LightGBM). Therefore, we have to find a way to encode (represent) these variables as numbers before handing them off to the model. There are two main ways to carry out this process:

# Label encoding: 
assign each unique category in a categorical variable with an integer. No new columns are created. An example is shown below image.png

# One-hot encoding: 
create a new column for each unique category in a categorical variable. Each observation recieves a 1 in the column for its corresponding category and a 0 in all other new columns. 

The problem with label encoding is that it gives the categoris an arbitary ordering. The value assigned to each of the categories is random and does not reflect any inherent aspect of the category. For more than 2 unique categories, one-hot encoding is the safe option.

The only downside to one-hot encoding is that the number of features (dimensions of the data) can explode with categorical variables with many categories. To deal with this, we can perform one-hot encoding followed by PCA or other dimensionality reduction methods to reduce the number of dimensions (while still trying to preserve information). For label encoding, use Scikit-Learn LabelEncoder and for one-hot encoding, use pandas get_dummies(df).

# Principal Component Analysis (PCA):
It is a dimensionality reduction technique widely used in machine learning and statistics. Its primary goal is to transform high-dimensional data into a lower-dimensional form while retaining as much of the original data's variability as possible. Here's a brief overview of PCA:

# Key Concepts:
1. Variance and Covariance:
PCA works by identifying the directions (principal components) along which the data varies the most. It relies on the covariance matrix, which captures the relationships between different features. 

2. Eigenvalues and Eigenvectors:
PCA calculates the eigenvalues and eigenvectors of the covariance matrix. Eigenvalues represent the amount of variance retained in each principal component, and eigenvectors represent the direction of maximum variance.

3. Principal Components:
Principal components are linear combinations of the original features. The first principal component accounts for the maximum variance, the second principal component (orthogonal to the first) accounts for the next maximum variance, and so on.

4. Explained Variance:
PCA provides a measure of how much of the total variance in the original data is retained in the reduced dimensions. By looking at the cumulative explained variance, one can decide on the number of principal components to retain.

# Steps in PCA:

1. Standardization:
Standardize the data by subtracting the mean and dividing by the standard deviation for each feature.

2.Covariance Matrix:
Calculate the covariance matrix for the standardized data.

3. Eigenvalues and Eigenvectors:

4. Sort and Select Principal Components:
Sort the eigenvalues in descending order and choose the top k eigenvectors (principal components) corresponding to the highest eigenvalues.

5. Project Data onto New Space:
Form a new matrix using the selected eigenvectors and project the original data onto this reduced-dimensional space.

