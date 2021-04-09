{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import scipy as sp \n",
    "import os\n",
    "import sklearn.preprocessing\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler\n",
    "from acquire import get_diabetic_data, get_id_data\n",
    "\n",
    "\n",
    "pd.set_option('display.max_columns', 80)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_diabetic_data():\n",
    "    '''This function takes in the dataframe, drops unneeded columns, makes clean dummy variables and handles nans.'''\n",
    "    df= get_diabetic_data()\n",
    "    #dropping unneeded columns\n",
    "df=df.drop(columns=['admission_type_id','patient_nbr', 'diag_1','diag_2', 'diag_3', 'readmitted', 'admission_source_id', 'discharge_disposition_id', 'payer_code',\n",
    "'change', 'medical_specialty', 'num_lab_procedures', 'num_procedures', 'number_emergency','number_outpatient','number_inpatient','number_diagnoses', 'diabetesMed'])\n",
    "    #replacing ? with nan\n",
    "    df['weight']=df['weight'].replace('?', np.nan)\n",
    "    df['race']=df['race'].replace('?', np.nan)\n",
    "    df['gender']=df['gender'].replace('?', np.nan)\n",
    "    #dropping nan\n",
    "    df = df.dropna()\n",
    "    #making dummy variables\n",
    "    dummy_df = pd.get_dummies(df[['race', 'gender','age','weight','metformin',\n",
    "       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',\n",
    "       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',\n",
    "       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',\n",
    "       'tolazamide', 'examide', 'citoglipton', 'insulin',\n",
    "       'glyburide-metformin', 'glipizide-metformin',\n",
    "       'glimepiride-pioglitazone', 'metformin-rosiglitazone',\n",
    "       'metformin-pioglitazone']])\n",
    "    #appending dummy variables to dataframe\n",
    "    df = pd.concat([df, dummy_df], axis=1)\n",
    "    #dropping non dummy columns\n",
    "    df= df.drop(columns=['metformin',\n",
    "       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',\n",
    "       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',\n",
    "       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',\n",
    "       'tolazamide', 'examide', 'citoglipton', 'insulin',\n",
    "       'glyburide-metformin', 'glipizide-metformin',\n",
    "       'glimepiride-pioglitazone', 'metformin-rosiglitazone',\n",
    "       'metformin-pioglitazone','num_medications','max_glu_serum','A1Cresult', 'race', 'gender','age', 'weight' ])\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def split_data():\n",
    "    '''this function takes in the dataframe and splits it into 3 samples, \n",
    "    a test, which is 20% of the entire dataframe, \n",
    "    a validate, which is 24% of the entire dataframe,\n",
    "    and a train, which is 56% of the entire dataframe.'''\n",
    "    df= clean_diabetic_data()\n",
    "    train_validate, test = train_test_split(df, test_size=.30, random_state=123)\n",
    "    train, validate = train_test_split(train_validate, test_size=.20, random_state=123)\n",
    "    return train, validate, test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "    def data_split(df, stratify_by='time_in_hospital'):\n",
    "    '''\n",
    "    this function takes in a dataframe and splits it into 3 samples, \n",
    "    a test, which is 20% of the entire dataframe, \n",
    "    a validate, which is 24% of the entire dataframe,\n",
    "    and a train, which is 56% of the entire dataframe. \n",
    "    It then splits each of the 3 samples into a dataframe with independent variables\n",
    "    and a series with the dependent, or target variable. \n",
    "    The function returns 3 dataframes and 3 series:\n",
    "    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. \n",
    "    '''\n",
    "    # split df into test (20%) and train_validate (80%)\n",
    "    train_validate, test = train_test_split(df, test_size=.2, random_state=123)\n",
    "\n",
    "    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)\n",
    "    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)\n",
    "    # split train into X (dataframe, drop target) & y (series, keep target only)\n",
    "    X_train = train.drop(columns=['time_in_hospital'])\n",
    "    y_train = train['time_in_hospital']\n",
    "    \n",
    "    # split validate into X (dataframe, drop target) & y (series, keep target only)\n",
    "    X_validate = validate.drop(columns=['time_in_hospital'])\n",
    "    y_validate = validate['time_in_hospital']\n",
    "    \n",
    "    # split test into X (dataframe, drop target) & y (series, keep target only)\n",
    "    X_test = test.drop(columns=['time_in_hospital'])\n",
    "    y_test = test['time_in_hospital']\n",
    "    \n",
    "    return X_train, y_train, X_validate, y_validate, X_test, y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def select_kbest(X_train, y_train, k):\n",
    "    '''\n",
    "    Takes in the predictors (X_train_scaled), the target (y_train), \n",
    "    and the number of features to select (k) \n",
    "    and returns the names of the top k selected features based on the SelectKBest class\n",
    "    '''\n",
    "    f_selector = SelectKBest(f_regression, k)\n",
    "    f_selector = f_selector.fit(X_train, y_train)\n",
    "    X_train_reduced = f_selector.transform(X_train)\n",
    "    f_support = f_selector.get_support()\n",
    "    f_feature = X_train.iloc[:,f_support].columns.tolist()\n",
    "    return f_feature"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def rfe(X_train, y_train, k):\n",
    "    '''\n",
    "    Takes in the predictor (X_train_scaled), the target (y_train), \n",
    "    and the number of features to select (k).\n",
    "    Returns the top k features based on the RFE class.\n",
    "    '''\n",
    "    lm = LinearRegression()\n",
    "    rfe = RFE(lm, k)\n",
    "    # Transforming data using RFE\n",
    "    X_rfe = rfe.fit_transform(X_train, y_train)\n",
    "    #Fitting the data to model\n",
    "    lm.fit(X_rfe,y_train)\n",
    "    mask = rfe.support_\n",
    "    rfe_features = X_train.loc[:,mask].columns.tolist()\n",
    "    return rfe_features"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
