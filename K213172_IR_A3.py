import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import streamlit as st
import os
import re
from nltk.stem import PorterStemmer
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import string

st.title("IR Assignment A3 Classification and Clustering (21K-3172)")
st.subheader("Classification")
#opening data set
df = pd.read_csv('data.csv', encoding='latin1')
X = df.drop('documentID', axis=1)
print(X.shape)

#labeling documents accordingly
y_dict = {
    1: "Explainable Artificial Intelligence",
    2: "Explainable Artificial Intelligence",
    3: "Explainable Artificial Intelligence",
    7: "Explainable Artificial Intelligence",
    8: "Heart Failure",
    9: "Heart Failure",
    11: "Heart Failure",
    12: "Time Series Forecasting",
    13: "Time Series Forecasting",
    14: "Time Series Forecasting",
    15: "Time Series Forecasting",
    16: "Time Series Forecasting",
    17: "Transformer Model",
    18: "Transformer Model",
    21: "Transformer Model",
    22: "Feature Selection",
    23: "Feature Selection",
    24: "Feature Selection",
    25: "Feature Selection",
    26: "Feature Selection"
}

df['labels']=df['documentID'].map(y_dict)
Y = df['labels']

df.head()
df['documentID'].unique()
x = df.drop(['documentID','labels'],axis=1)
y = df['labels']
y = pd.get_dummies(y)

print("Missing values in x:", x.isnull().sum())
print("Missing values in y:", y.isnull().sum())


#Opening files function
def LocateFile(fileLocation):#ResearchPapers/1.txt
  fileContent=""
  if os.path.exists(fileLocation):
    with open(fileLocation,'r',encoding='latin-1') as f:
      fileContent=f.read()
    return fileContent
  else:
    return ""
  
#Preprocessing
def Preprocessing(fileContent, stopWords):
  tokens = word_tokenize(fileContent) #run the tokenizer for the words
  uselessWords=['by','me','etc','or','but','not','and']#additional list of words
  i=0
  #to go over the terms and run basic modification
  while i<len(tokens):#for case folding+puchtuation
    token=tokens[i]
    token=token.lower()#case fold
    #split words with puctuation into mulitiple words and include them in the tokken set
    words = re.sub('['+string.punctuation+']', ' ', token)#replace puctutation with space
    words=words.split(' ')#break into words and add to the token lists at the right pace
    tokens.pop(i)
    tokens=tokens[:i]+words[:]+tokens[i:]#put them in the right position to maintian proximity index
    i+=1


  i=0
  #get rid of the numbers and deal with words with numbers by breaking into smaller words
  while i<len(tokens):
    token=tokens[i]
    #handle words with numbers/numbers
    if re.search(r'\w*\d\w*', token):  #the token contains any numbers
      word=""
      words=[]
      k=0
      for j in token:#split into smaller words for further processing
        if(not j.isdigit()):#store words and discard numbers
          word+=j
        elif (word!=""):
          words.append(word)
          word=""
          k+=1

      if(word!=""):#make sure there isnt a word in temp variable
        words.append(word)
        k+=1
     
      tokens.pop(i)#delete the current token to replace with the new words
      tokens=tokens[:i]+words[:]+tokens[i:]#add the words in the correct place to maintain position
      i+=k
      # tokens.extend(words)
      continue
    else:
      tokens[i]=token
      i+=1

  i=0
  while i<len(tokens):
    token=tokens[i]
    #handle words with special characters
    if re.search(r'\w*[!@#\$%^&*()_+\-=\[\]{};:"\\|,.<>\/?]\w*', token):#the word contains a special char
      token = re.sub(r'[!@#\$%^&*()_+\-=\[\]{};:"\\|,.<>\/?]', ' ', token)#remove the char
      words=token.split(' ')#split into smaller words
      tokens.pop(i)
      tokens=tokens[:i]+words[:]+tokens[i:]#put them in the right position to maintian proximity index
    if re.search(r'\w*[\x91-\x99]\w*', token):#do the same for escape character?(like quotes etc)
      words = re.sub(r'[\x91-\x99]', ' ', token)
      words=words.split(' ')
      tokens.pop(i)
      tokens=tokens[:i]+words[:]+tokens[i:]
    if re.search(r'\w*[\x9a\x8e\x88\x8a]\w*',token):#special case escap characters
      words = re.sub(r'[\x9a\x8e\x88\x8a]',' ',token)
      words=words.split(' ')
      tokens.pop(i)
      tokens=tokens[:i]+words[:]+tokens[i:]
    i+=1


  #after modifiying the the tokens we run them though the filter
  cleanedTokens={}
  index=0#to maintain the position
  for token in tokens:#process the tokens/clean
    flag=False
    #the token is a stop word or custom made useless word
    for stopword in stopWords: 
      if(stopword==token):
        flag=True
        break
    for uselessword in uselessWords:
      if(uselessword==token):
        flag=True
        break

    if(flag):#the word belongs to either of the mentioned lists
      index+=1
      continue
    
    if (len(token)<=1): #the token is a single charachter or empty string(special case/exception)
      continue
   
    #if it passes all previous filters then place it in the dictionary
    # print(token)
    if(token in cleanedTokens):#already present in the dictionary
      cleanedTokens[token]+=1
    else:
      cleanedTokens[token]=1
    index+=1
  return cleanedTokens

#processing query
def QueryProcessing(query):
        ps = PorterStemmer()#init porter stemmer of index(if we cant load it) and query
        stopWords= LocateFile('Stopword-List.txt') #get a preset stopword list
        stopWords=word_tokenize(stopWords) #toeknize the stopwords

        cleanedWords=Preprocessing(query,stopWords)
        words=[]
        for key in cleanedWords.keys():
            word=ps.stem(key)
            for i in range(cleanedWords[key]):
                words.append(word)
        return words

#Creating Query Vector
def QueryVector(query, termArray,idf):
        terms=QueryProcessing(query)
        result=[0 for i in range(len(termArray))]

        for term in terms:
            if(term in termArray):
                index=termArray.index(term)
                result[index]+=1

        for term in terms:
            weight=0
            if(term in termArray):
                index=termArray.index(term)
                result[index]*=idf[term]
                weight+=(result[index]**2)

        weight=weight**0.5
        for i in range(len(result)):
            if(result[i]!=0):
                result[i]/=weight

        return result


termArray=None
idf=None
# with open('./DocTermVector.pickle','rb') as f:
#     docTermVectors=pickle.load(f)
#Loading Vecotor Space Dataset
with open('term_array.pickle','rb') as f:
    termArray=pickle.load(f)
with open('scores_idf.pickle','rb') as f:
    idf=pickle.load(f)

#Using Knn for classification
knn = KNeighborsClassifier()
knn.fit(X,Y)
y_pred_train = knn.predict(X)
class_names = knn.classes_
print(f"Accuracy on Training data: {accuracy_score(Y,y_pred_train)}\n")
precision_scores = precision_score(Y, y_pred_train, average=None)
recall_scores = recall_score(Y, y_pred_train, average=None)
f1_scores = f1_score(Y, y_pred_train, average=None)
# print(f'Precision: {precision_scores}')
# print(f'Recall: {recall_scores}')
# print(f'F1 : {f1_scores}')
st.write(f"Accuracy on Training data: {accuracy_score(Y,y_pred_train)}\n")
for i in range(len(class_names)):
    st.write(f'Class: {class_names[i]}')
    st.write(f'Precision: {precision_scores[i]}')
    st.write(f'Recall: {recall_scores[i]}')
    st.write(f'F1 Score: {f1_scores[i]}')
    st.write('\n')  # Add an empty line between each class metrics


#testing data
# print('===============================')
# para="""Heart failure, also known as congestive heart failure, is a chronic condition where the heart is unable to pump enough blood to meet the body's needs. This can lead to symptoms such as shortness of breath, fatigue, swelling in the legs, ankles, or abdomen, and rapid or irregular heartbeat. There are several types of heart failure, including left-sided, right-sided, systolic, and diastolic heart failure, each with its own causes and characteristics. Treatment for heart failure typically involves medications to help the heart pump more effectively, lifestyle changes such as diet and exercise, and in some cases, surgical interventions. Managing heart failure requires a comprehensive approach that addresses both the physical and emotional aspects of the condition, and regular monitoring by healthcare professionals is essential to ensure the best possible outcomes for patients."""

# para2 = """
# Explainable Artificial Intelligence (XAI) is a crucial field focused on making AI systems and their decisions understandable and transparent to humans. XAI techniques aim to provide insights into the inner workings of AI models, allowing users to comprehend why a particular decision was made. This transparency is vital, especially in high-stakes applications like healthcare and finance, where AI decisions can have significant impacts. XAI approaches include using interpretable models, determining feature importance, providing local explanations for individual predictions, promoting human-AI collaboration, and using visualizations to represent complex decision-making processes. By incorporating XAI, AI systems can be more accountable, fair, and trustworthy, fostering greater acceptance and adoption in various domains.
# """
# para3= """
# Transparency and interpretability in artificial intelligence (AI) are essential for building trust and understanding in AI systems. Transparent AI refers to the ability to understand the logic, process, and outcomes of AI models, ensuring that users can trust the decisions made by these systems. Interpretable AI focuses on making the internal workings of AI models understandable to humans, enabling users to comprehend how inputs are processed to produce outputs. By promoting transparency and interpretability, AI systems become more accountable, allowing users to verify the fairness, accuracy, and ethical considerations of these systems. This transparency also encourages collaboration between humans and AI, facilitating more effective and trustworthy decision-making processes across various domains.
# """
# para4 = """
# This is a crucial aspect of machine learning and data mining research, aimed at identifying the most relevant variables or features from a dataset to improve model performance and interpretability. This process involves selecting a subset of features that are most informative for the target variable, while excluding irrelevant or redundant ones. Research in feature selection encompasses various techniques, including filter methods that rank features based on statistical measures, wrapper methods that use the model's performance as a criterion, and embedded methods that integrate feature selection into the model's training process. Recent advancements in this field focus on developing efficient algorithms for high-dimensional datasets, handling missing data, and addressing the challenges posed by complex data structures such as time series or text data. Feature selection research plays a pivotal role in enhancing the efficiency, accuracy, and interpretability of machine learning models across diverse application domains.
# """
# para5 = """
# Analyzing sequential data is crucial for understanding trends and making informed decisions. By examining past data points, patterns, and trends, businesses can anticipate changes and adapt strategies. This analysis helps organizations predict demand, identify seasonality, and manage risk. In finance, this approach aids in predicting stock prices, currency exchange rates, and economic indicators. Weather forecasting also benefits from this method, predicting future conditions based on historical data. Overall, analyzing sequential data is essential for data-driven decision-making, operational optimization, and competitive advantage across industries.
# """
# para6 = """
# Transformer are translating text and speech in near real-time, opening meetings and classrooms to diverse and hearing-impaired attendees. They’re helping researchers understand the chains of genes in DNA and amino acids in proteins in ways that can speed drug design.
# """
#Entering Text For Classification
input_text = st.text_input("Enter Text for Classification")
if input_text:
  queryVector=QueryVector(input_text,termArray,idf)
  # np.reshap()
  queryVector = np.array(queryVector).reshape(1, -1)
  result=knn.predict(queryVector)
  st.write(f'Class: {result[0]}')

#Clustering

st.subheader("Clustering")
# Assuming df is your dataframe
X = df.drop(['documentID','labels'], axis=1)  # Remove documentID column

# Perform KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=45)  # You can choose the number of clusters
df['cluster'] = kmeans.fit_predict(X)
# df[['labels','documentID', 'cluster']]
# Group by cluster and labels, then aggregate documentID into a list
cluster_class_mapping = df.groupby(['cluster', 'labels'])['documentID'].apply(list).reset_index()
print(cluster_class_mapping)
st.write(cluster_class_mapping )
from sklearn import metrics
# Compute Purity
true_labels = df['labels']

#Computing metrics scores for evaluation
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

purity = purity_score(true_labels, df['cluster'])
st.write("Purity Score: ", purity)

# Compute Silhouette Score
silhouette_score = metrics.silhouette_score(X, df['cluster'])
st.write("Silhouette Score: ", silhouette_score)

# Compute Rand Index
rand_index = metrics.adjusted_rand_score(true_labels, df['cluster'])
st.write("Rand Index: ", rand_index)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Plot the clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clustering Results')
plt.colorbar(scatter, label='Cluster')

# Display plot in Streamlit
st.pyplot(plt)


input_text_cluster = st.text_input("Enter Text for Identifying Cluster")

if input_text_cluster:
  queryVector=QueryVector(input_text_cluster,termArray,idf)
  # np.reshap()
  queryVector = np.array(queryVector).reshape(1, -1)
  result=kmeans.predict(queryVector)
  cluster_labels = kmeans.labels_
  # st.write("Cluster Labels:", cluster_labels)
  st.write("Predicted Cluster for New Text:", result[0])