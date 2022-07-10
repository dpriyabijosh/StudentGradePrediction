#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# 
# **Objective:** Understanding the data and find the pattern and trends about the data
# 
# **Data Set Information**
# The data used is from a Portuguese secondary school. The data includes academic and personal(Economic factors) characteristics of the students as well as final grades. The task is to predict the final grade from the student information. (Regression)
# [Link to dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)

# # Import Libraries

# In[4]:


# Importing required libraries.
import pandas as pd # Importing pandas
import numpy as np # Importing numpy
import seaborn as sns #Importing Seaborn for data visualisation
import matplotlib.pyplot as plt #Importing matplotlib for data visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# # The Data 
# The data is added to 2 data frames studentMath and Student Port

# In[5]:


# Function to fetch the data set
# @path : path of the data file
def fetchDataSet(path):
    dataSet =""
    try:
        dataSet = pd.read_csv(path,sep=';')
    except FileNotFoundError:
        print("The file is not found!!. Please add a valid file in the given path.")
    return dataSet  


# In[6]:


#Function to get both the datasets
def getDatasets():
    studentMath = fetchDataSet('DataFile/student-mat.csv')#StudentMath contains the data of student data on Mathematics
    studentPort = fetchDataSet('DataFile/student-por.csv')#StudentPort contains the data of student data on Portuguese 
    return studentMath,studentPort


# Checking the size and columns of the 2 datasets

# In[7]:


#Function to get the dataset
def dasetInfo(name,dataset):
        print("Checking the shapes of the {} data set :".format(name),dataset.shape)
        print("Columns\n",dataset.columns)


# # Observations
# Maths dataset have 395 observations while Portuguese dataset have 649 observations.
# Both data sets have same 33 columns.

# Final data set is created by concantenating the data sets. Also,the duplicate entries are removing using the similiar attributes.

# In[8]:


# Merging the given datasets and removing the duplicates based on the similiar attributes
# @dataSets : the datasets to concantanate
# @sim-attr : Similiar attributes to find duplicates
def mergeDataSets(dataSets,sim_attr):
    data = pd.concat(dataSets)
    finalDataSet = data.drop_duplicates(subset=sim_attr, keep ='first').reset_index(drop=True)#keep = first keeps the values of first joined dataset
    return finalDataSet


# In[9]:


#Function to get the dataset
def getDataSet():
    studentMath, studentPort = getDatasets()
    studentDataSet = mergeDataSets([studentMath, studentPort],['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','nursery','internet'])
    return studentDataSet
studentDataSet = getDataSet()


# We got the final dataset with 662 observations by removing 382 duplicate entries.
# Data has categorical and integer values.

# Checking the null values 

# Data has categorical and integer values. There is no need of data cleansing since the dataset does not have any missing/null values.

# In[10]:


#Function to datamining and getting information about the data
def dataInfo():
    studentMath = fetchDataSet('DataFile/student-mat.csv')#StudentMath contains the data of student data on Mathematics
    studentPort = fetchDataSet('DataFile/student-por.csv')#StudentPort contains the data of student data on Portuguese 
    print('Checking the info of Maths dataset',dasetInfo("Maths",studentMath))
    print('Checking the info of Maths dataset',dasetInfo("Portuguese",studentPort))
    print('Merging the 2 datasets')
    studentDataSet = mergeDataSets([studentMath, studentPort],['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','nursery','internet'])
    # Checking the details of the dataset
    print('The shape of the dataset is :', studentDataSet.shape,'\nThe dataset description\n')
    print('The details of the dataset\n',studentDataSet.info())
    print('Checking the null values of the dataset\n')
    print(studentDataSet.isnull().sum())
    print('Observe first 5 entries of the dataset\n',studentDataSet.head())


# # Summary 

# We have calculated the statistics functions of all numerical values and created a statistics table
# The categorical variables relations and interpretations have described using plots

# In[47]:


# Importing library for skew and kurtosis
from scipy.stats import skew
from scipy.stats import kurtosis

class SummaryStatistics():#SummaryStatistics inherits StatisticsClass Class
    def __init__(self, data_frame):
        self.__dataframe = data_frame
    def getMean(self):
        return np.array(round(np.mean(self.__dataframe),4))
    def getStd(self):
        return np.array(round(np.std(self.__dataframe),4))
    def getMinimum(self):
        return np.array(round(np.min(self.__dataframe),4))
    def getLowerQuadrile(self):
        return np.array(np.percentile(self.__dataframe,25, axis=0))
    def getMedian(self):
        return np.array(np.percentile(self.__dataframe,50, axis=0))
    def getUpperQuadrile(self):
        return np.array(np.percentile(self.__dataframe,75, axis=0))
    def getMaximum(self):
        return np.array(round(np.max(self.__dataframe),4))
    def getKurtosis(self,bias=False,fisher=True,axis=0):
        #calculate sample kurtosis
        return kurtosis(self.__dataframe,axis,fisher, bias)
    def getSkew(self,bias=False,axis=0):
        return skew(self.__dataframe,axis, bias)
    def getStatistics(self):
        finalmatrix= np.concatenate([self.getMean(),self.getStd(),self.getMinimum(),self.getLowerQuadrile(),self.getMedian(),self.getUpperQuadrile(),self.getMaximum(),self.getSkew(),self.getKurtosis()], axis=0)
        statValues = ['Mean','Std','Minimum','First quartile','Median','Third quartile','Maximum','Skew','Kurtosis']
        Variables = ['Age','Mother education','Father education','Travel time','Study time','Failures','Family relation','Free time','Go out','Daily alcohol','Weekly alcohol','health','absences','G1','G2','G3']
        df2 = pd.DataFrame(data= finalmatrix.reshape(9,16), index=statValues,columns=Variables)
        return df2.transpose()


# In[48]:


def statistics(studentDataSet):#Main function. to call the statistics function
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    selectVariables = studentDataSet.select_dtypes(include=numerics)
    stat = SummaryStatistics(selectVariables)
    global stat_table #Declared globally to use outside to see as a dataframe
    stat_table = stat.getStatistics()
    return stat_table


# In[49]:



 # Observations:
# 
# There are 662 of data entries with only two distinct schools.
# The percentage of female to male is 58% and 42% respectively
# The average age of a student is 17 years with 15 years minimum and 22 years maximum.
# A student may averagely,absent hiself/herself 4 days from school.
# The total number of features are 34 including the added subject while 16 is integer type and 18 string type

# # Data Visualization

# In[1]:


#Final grade distribution
def plotFinalGrade(x):
    plt.figure(figsize=(12,6))
    plt.title('Final Grade Distribution',fontdict={'fontname':'arial', 'fontsize':17})
    sns.histplot(x=x['G3'], color='b', bins=10)
    plt.show()


# In[29]:


# Histogram of all variables
def ditribution(data):
    fig = plt.figure(figsize = (10,10))
    ax = fig.gca()
    data.hist(ax=ax)   


# In[20]:


#Gender distribution
def genderDistributionPlot(dataset):
    sns.set_style('whitegrid')    # male & female student representaion on countplot
    b= sns.countplot(x='sex',data=dataset,palette='plasma')
    b.axes.set_title('Gender distribution')
    x_labels = ['Female','Male']
    b.set_xticklabels(x_labels)
    plt.show()


# The number of females are higher than the male students

# # Students' Age

# In[52]:


#Student age distribution
#@data is the dataset
def studentAgeDistribution(data):
    plt.figure(figsize=(12,6))
    b = sns.kdeplot(data['age'])    # Kernel Density Estimations
    b.axes.set_title('Ages of students',fontdict={'fontname':'arial', 'fontsize':17})
    b.set_xlabel('Age', fontsize = 15)
    b.set_ylabel('Count', fontsize = 15)
    plt.show()


# In[53]:


#Student age distribution with gender
#@data is the dataset
def studentAgeDistriWithGender(data):
    plt.figure(figsize=(12,6))
    b = sns.countplot(x='age',hue='sex', data=data, palette='inferno')
    b.axes.set_title('Number of Male & Female students in different age groups',fontdict={'fontname':'arial', 'fontsize':17})
    plt.legend(loc='best')
    b.set_xlabel("Age",fontsize = 15)
    b.set_ylabel("Count", fontsize = 15)
    plt.show()


# The student age seems to be ranging from 15-19, where gender distribution is pretty even in each age group.
# The age group above 19 may be year back students or droupouts.

# # Students from Urban & Rural Areas

# In[54]:


#Student location distribution
#@data is the dataset
def studentDistributionAreas(data):
    plt.figure(figsize=(12,8))
    b=sns.countplot(x='address',hue='G3',data=data,palette='Oranges')
    b.axes.set_title('Number of Students based on location',fontdict={'fontname':'arial', 'fontsize':17})
    x_labels = ['Urban','Rural']
    b.set_xlabel("Location", fontsize = 15)
    b.set_ylabel("Count", fontsize = 15)
    b.set_xticklabels(x_labels,fontsize=15)
    plt.show()


# Approximately 70% students come from urban region and 30% from rural region.

# # Checking the dependencies of socio-economic factors

# 1. Does age affects the score?

# In[55]:


#Function to check the impact of age on final score
#@data is the dataset
def ageVsGrade(data):
    plt.figure(figsize=(12,8))
    b= sns.boxplot(x='age', y='G3',data=data,palette='gist_heat')
    b.axes.set_title('Age vs Final Grade',fontdict={'fontname':'arial', 'fontsize':17})
    plt.show()


# The above plot shows that the median grades of the three age groups(15,16,17) are similar. 
# Also, 19 and 21 have the same median and also very small spread compared to others.
# The age group 15,16,17,19 have outliers of score 0 wither may be due to absence or very low scored students with 0 marks.
# Age group 20 seems to score highest grades among all.

# 2. Do urban students perform better than rural students?

# In[56]:


#Function to check the impact of location on final score
#@data is the dataset
def distrAddress(data):
    plt.figure(figsize=(13,8))
    sns.kdeplot(data.loc[data['address'] == 'U', 'G3'], label='Urban', shade = True)
    sns.kdeplot(data.loc[data['address'] == 'R', 'G3'], label='Rural', shade = True)
    plt.title('Do urban students score higher than rural students?',fontdict={'fontname':'arial', 'fontsize':17})
    plt.legend(loc='best')
    plt.xlabel('Grade',fontsize=15);
    plt.ylabel('Density', fontsize = 15)
    plt.show()


# From the above graph we can understand that there is not much difference between the grades based on location.

# 3. Do the past failures have high corelation on the final grade  G3?

# In[57]:


#Function to check the impact of previous failures on final score
#@data is the dataset
def failuresDistnSex(data):
    sns.catplot(x="failures", y="G3", hue="sex",
                kind="violin", split=True, data=data)
    plt.title('No of past failures vs Final score',fontdict={'fontname':'arial', 'fontsize':17})
    plt.xlabel('Number of past failures',fontsize=15);
    plt.ylabel('Final score', fontsize=15)
    plt.show()


# Number of past failures have significant effects on student final grade. The students which have 0 failures made high scores than other students

# # Family Attributes

# # Parent's job

# In[58]:


# Define Boxplot function, to plot all three exam scores for different variables, check median and quartiles of the scores.
# plt.figure(figsize=(16,10))
# plt.show()
def boxpl(dt, x_cols, y_cols,title='Title'):
    n = 1
    x_cnt = len(x_cols)
    y_cnt = len(y_cols)
    figure = plt.figure(figsize=(17, 5 * x_cnt))
    figure.suptitle(title, fontsize=20)
    for x_ax in x_cols:
        for i in y_cols:
            ax = figure.add_subplot(x_cnt, y_cnt, n)
            #ax.set_title(i)
            g = sns.boxplot(x = dt[x_ax], y = dt[i])
            g.set_xticklabels(g.get_xticklabels(), rotation=20)
            n = n + 1         


# From this plot, we can understand that parents job have impact on students score. The students with mother/ father as teachers or in health sector scores more marks.
# The students with other/at home have achieved low marks compared to others.

# # Parent's Education

# In[59]:


#The chart styled using ggplot
from matplotlib import style
def subBoxplot(title,x_value,x1_value,y_value,label1,label2,x_order):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(title, fontsize=30)
        sns.boxplot(x=x_value, y=y_value, data=studentDataSet, ax=ax1)
        ax1.title.set_text(label1)
        #Checking with father's education
        var1 = ax1.set_xticklabels(x_order, rotation=30)
        sns.boxplot(x=x1_value, y=y_value, data=studentDataSet, ax=ax2)
        ax2.title.set_text(label2)
        var2 = ax2.set_xticklabels(x_order, rotation=30)


# Students with high educated parents score more than other students.
# But there are some students who scores well even though there parents have no education.

# In[60]:


#Function to check the impact of guardian on final score
#@data is the dataset
def guardianvsG3(data):
    b = sns.catplot(x="guardian", y="G3", data=data, hue='Pstatus',palette='plasma')
    plt.title('Family size and Grade by classifying the parent\'s cohabitation status', fontdict={'fontname':'arial', 'fontsize':15})
    plt.ylabel('Final score', fontsize=15)
    plt.xlabel('Guardian',fontsize=15)


# The students who lives with their mother are achieving more scores than others even though the parents are apart.
# But the students who lives with father achieving more scores but from this plot, we can understand that both parents are living together.
# In the case of students lives with others does not affects much about the parents cohabitation status

# In[61]:


#Function to check the impact of family and schools support on final score
#@data is the dataset
def supportvsGrade(data):
    sns.catplot(x='famsup',y='G3',data=data,kind="bar",hue='schoolsup',palette='plasma')
    plt.title('Family support vs Final Grade', fontdict={'fontname':'arial', 'fontsize':15})
    plt.ylabel('Final score', fontsize=15)
    plt.xlabel('Family Support',fontsize=15)
    plt.show() 


# Students getting both family support and school support became more lazy and score low scores than the students without any support.

# # Reason of selecting school

# In[79]:


#Function to check the distribution of the reason of selecting schools 
def DistribnReason(data):
    plt.figure(figsize=(11,7))
    sns.set(font_scale=1.5)
    sns.set_style('white')
    ax = sns.countplot(y='reason',data=data, order=data['reason'].value_counts().index, hue='school', palette="Blues_d")
    ax.set_yticklabels(('Course Preference', 'Close to Home', 'School Reputation', 'Other'))
    plt.ylabel('Reason')
    plt.xlabel('Count')
    plt.title('Reason For Choosing The School')
    plt.show()


# In[80]:


#Function to check the impact of reason on final score
#@data is the dataset
def reasonvsGrade(data):
    sns.set(font_scale=1.5)
    sns.set_style('white')
    sns.catplot(x="reason", y="G3", data=data)
    plt.ylabel('Final Grade')
    plt.show()


# Observation : The students have an equally distributed average score when it comes to reason attribute.

# # Relation with the study time and travelling time

# In[81]:


#Function to check the impact of time spent on final score
#@data is the dataset
def timeSpendgrade(data):
    sns.catplot(x='traveltime',y='G3',data=data,kind="bar",hue='studytime',palette='plasma')
    plt.title('Relation of time spend and marks awarded', fontdict={'fontname':'arial', 'fontsize':15})
    plt.ylabel('Final Grade')
    plt.show()   


# The travelling time have significant relation with the final grade. Still we can find out that the students who spends more time on studies can achieve high marks eventhough they have high travel time.

# Early Education and Future educationPlan

# In[82]:


#Function to check the impact of higherstudy on final score
#@data is the dataset
def higherStudy(data):
    sns.catplot(x='higher',y='G3',data=data,kind="box",hue='nursery',palette='cool')
    plt.title('Future Study plan vs Final(Grade)', fontdict={'fontname':'arial', 'fontsize':15})
    plt.ylabel('Final Grade')
    plt.show()


# Early education does not have much effects on the score awarded but their future study plan have very high impacts on the score.
# The students who are intented to so a higher education is achieving very high score.

# In[83]:


#Function to check the impact of internet on final score
#@data is the dataset
def internetvsFinalGrade(data):
    sns.catplot(x='internet',y='G3',data=data,kind="bar",hue='reason',palette='deep')
    plt.title('Internet and grade', fontdict={'fontname':'arial', 'fontsize':15})
    plt.ylabel('Final Grade')
    plt.show()


# # Personal Life

# In[84]:


#Function to check the impact of romantic relationship on final score
#@data is the dataset
def romanticvsGrade(data):
    sns.catplot(x='romantic',y='G3',data=data,kind="strip",palette='YlOrBr', hue_order=["small", "big"])
    plt.title('Romantic relationship and grade', fontdict={'fontname':'arial', 'fontsize':20})
    plt.ylabel('Final Grade')
    plt.show()


# Observation : Students with no romantic relationship score higher

# # Going out with friends

# In[85]:


#Function to check the distribution of going out
#@data is the dataset
def distrnGoOut(studentDataSet):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Medium','Low','High','Very High','Very Low'
    sizes = studentDataSet['goout'].value_counts()
    explode = (0, 0.1, 0, 0,0.1)  # "explode" the 2nd and 5th slice

    plt.pie(sizes, explode=explode, labels=labels, 
    autopct='%1.1f%%', shadow=True, startangle=140) # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Going out with friends percentage',fontdict={'fontname':'arial', 'fontsize':20})
    plt.show()


# Observation : The studentsshould have some social life.

# In[90]:


#Function to check the impact of going out on final score
#@data is the dataset
def goOutFinalScore(studentDataSet):
    sns.catplot(x='goout',y='G3',data=studentDataSet,kind="bar",palette='autumn')
    plt.title('Go Out vs Final Grade(G3)')
    plt.ylabel('Final Grade')
    plt.show()


# Observation : Students who go out a lot score less

# # Alcohol consumption

# In[91]:


#Function to check the distribution of alcohol
#@data is the dataset
def distrnAlcohol(studentDataSet):
    b = sns.countplot(x=studentDataSet['Walc'],palette='OrRd')
    labels = 'Very Low','Low','Medium','High','Very High'
    b.set_title('Daily alchohol consumption count',fontdict={'fontname':'arial', 'fontsize':20})
    b.set_xticklabels(labels,fontdict={'fontname':'arial', 'fontsize':15})
    b.set_xlabel('Daily alcohol consumption',fontdict={'fontname':'arial', 'fontsize':15})
    plt.show()


# In[92]:


#Function to check the impact of final score on final score
#@data is the dataset
def alcoholvsFinalScore(studentDataSet):
    sns.catplot(x='Walc',y='G3',data=studentDataSet,kind='bar',hue='sex',palette='plasma')
    plt.title('Weekly alcohol consumption vs Grade', fontdict={'fontname':'arial', 'fontsize':15})
    plt.show()


# When considering the weekly alcohol consumption, the less alcohol used students achieve high score than other  students. But when considering gender, the high alcohol used females achieving high score than the female students who consume less alcohol.

# # Health of the student

# In[93]:


#Function to check the impact of health on final score
#@data is the dataset
def healthVsScore(studentDataSet):
    sns.catplot(x='health',y='G3',data=studentDataSet,kind="bar",palette='muted')
    plt.title('Health vs Final Grade(G3)')
    plt.show()


# The students with very bad health conditions score better score

# In[94]:


#Function to check the iabsence distribution
#@data is the dataset
def absenceDistrbn(data):
    sns.countplot(x="absences", data=studentDataSet, color="c")
    plt.title('Absence distribution')
    plt.show()


# Absence rate are not very high. About 320 students does not taken any leaves.

# In[95]:


#Function to check the impact of absence on final score
#@data is the dataset
def absenceVsScore(data):
    sns.catplot(x='absences',y='G3',data=studentDataSet,kind="bar",palette='muted')
    plt.title('Health vs Final Grade(G3)')
    plt.show()


# In[96]:


#Function to do all the visualization as part of EDA
#@data is the dataset
def plotMain(dataset):
    ditribution(dataset)
    plotFinalGrade(dataset)
    genderDistributionPlot(dataset)
    studentAgeDistribution(dataset)
    studentAgeDistriWithGender(dataset)
    ageVsGrade(dataset)
    studentDistributionAreas(dataset)
    distrAddress(dataset)
    y_cols = ['G3']
    x_cols = ['Mjob', 'Fjob']
    boxpl(studentDataSet, x_cols, y_cols,'Parents Jobs vs final grade')
    plt.show()
    subBoxplot('Comparing Parents education vs Final grade awarded','Medu','Fedu','G3', 
           'Mother\'s education vs Grade','Father\'s education vs Grade',
           ('None', 'Primary education', '5th to 9th grade', 'Secondary education',"Higher education")) 

    guardianvsG3(dataset)
    supportvsGrade(dataset)
    y_cols = ['G3']
    x_cols = [ 'G1','G2']
    boxpl(dataset, x_cols, y_cols,'Previous Scores vs final grade')
    plt.show()
    guardianvsG3(dataset)
    DistribnReason(dataset)
    reasonvsGrade(dataset)
    timeSpendgrade(dataset)
    higherStudy(dataset)
    internetvsFinalGrade(dataset)
    romanticvsGrade(dataset)
    distrnGoOut(dataset)
    goOutFinalScore(dataset)
    distrnAlcohol(dataset)
    alcoholvsFinalScore(dataset)
    healthVsScore(dataset)
    absenceDistrbn(dataset)
    failuresDistnSex(dataset)
    y_cols = ['G3']
    x_cols = ['G1', 'G2']
    boxpl(studentDataSet, x_cols, y_cols,'Comparing Previous grades vs final grade')


# The G1 and G3 directly propotional to G3. The students have high previous grade score good marks in final grade too.

# # Correlation

# In[97]:


#Checking correlation between the class/target and the features
def checkCorrelation(studentDataSet):
    corr_matrix = studentDataSet.corr()
    print(corr_matrix["G3"].sort_values(ascending=False),"\n\n")
    # Set size of heatmap
    plt.figure(figsize=(20, 10)) 
    # Store heatmap object in a variable to easily access it.
    # Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
    heatmap = sns.heatmap(studentDataSet.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=12);

