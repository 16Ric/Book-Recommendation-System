import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def compute_probability(col):
    """
    Compute the probability of a certain event
    """
    return col.value_counts() / col.shape[0]

def compute_entropy(col):
    """
    Compute the entropy of a certain event
    """
    probabilities = compute_probability(col)
    entropy = -sum(probabilities * np.log2(probabilities))
    return entropy

def compute_conditional_entropy(x, y):
    """
    Compute the conditional entropy between two random variables.
    Specifically, the conditional entropy of Y given X.
    """
    probability_x = compute_probability(x)
    
    temp_df = pd.DataFrame({'X': x, 'Y': y})
    
    conditional_entropy = 0
    
    # for unique event x_i
    for x_i in x.unique():
        # get the data for Y given X=x_i
        y_given_x = temp_df.loc[temp_df['X'] == x_i, 'Y']
        
        # compute the conditional entropy
        conditional_entropy += probability_x[x_i] * compute_entropy(y_given_x)
    
    return conditional_entropy

def compute_information_gain(x, y):
    """
    Compute the information gain between an attribute and class label
    """
    return compute_entropy(y) - compute_conditional_entropy(x, y)

def bin_age(x):
    """
    Domain knowledge to bin the age
    """
    UPPER1 = 18
    UPPER2 = 24
    UPPER3 = 29
    UPPER4 = 39
    UPPER5 = 49
    UPPER6 = 64
    
    if x < UPPER1:
        return 0
    elif UPPER1 < x <= UPPER2:
        return 1 
    elif UPPER2 < x <= UPPER3: 
        return 2
    elif UPPER3 < x <= UPPER4:
        return 3
    elif UPPER4 < x <= UPPER5:
        return 4
    elif UPPER5 < x <= UPPER6:
        return 5
    elif UPPER6 < x:
        return 6
    return 7

def bin_rating(x):
    """
    Domain knowledge to bin the rating
    """
    LOWER = 4
    MEDIUM = 7

    if x <= LOWER:
        return 0
    elif LOWER < x <= MEDIUM:
        return 1 
    return 2

def euclidean_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    
    # Removing punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Joining tokens back into text
    text = ' '.join(tokens)
    
    return text

# Merge the given dataset
bxBooks = pd.read_csv('BX-Books.csv')
bxRatings = pd.read_csv('BX-Ratings.csv')
bxUsers = pd.read_csv('BX-Users.csv')
bxBooksRating = pd.merge(bxBooks, bxRatings, on='ISBN', how='left')
bxBooksUserRating = pd.merge(bxBooksRating, bxUsers, on='User-ID', how='left')

# Bin Rating
bxBooksUserRating['binned_rating'] = bxBooksUserRating['Book-Rating'].apply(lambda x: bin_rating(x))

# Cleaning the User-Age with '\"'
age_pattern = r'\d+"'
for index, age in bxBooksUserRating["User-Age"].items():
    if re.search(age_pattern, str(age)):
        cleaned_age = re.sub(r'"', '', str(age).strip()).lower()
        bxBooksUserRating.at[index, "User-Age"] = cleaned_age

# Age Outliers Removal
bxBooksUserRating['Book-Rating'] = pd.to_numeric(bxBooksUserRating['Book-Rating'], errors='coerce')
bxBooksUserRating['User-Age'] = pd.to_numeric(bxBooksUserRating['User-Age'], errors='coerce')

#----- Pie Chart NaN values
#bxBooksUserRating['nan_age'] = bxBooksUserRating['User-Age'].isna()
#nan_cluster = bxBooksUserRating.groupby('nan_age').size()

#plt.figure(figsize=(3, 3))
#plt.pie(nan_cluster, labels=bxBooksUserRating.groupby('nan_age').size().index, autopct='%1.1f%%', startangle=140)
#plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
#plt.title('Percentage of NaN Age')
# Save the plot as a PNG file
#plt.savefig('nan_age_pie_chart.png')

# Show the plot
#plt.show()

#---- NaN Values Size
#print(nan_cluster)
#-----

age_outliers = (bxBooksUserRating["User-Age"] < 6) | (bxBooksUserRating["User-Age"] > 100)
bxBooksUserRating["User-Age"] = bxBooksUserRating["User-Age"][~age_outliers]

#----- MEDIAN
#value_counts = bxBooksUserRating['User-Age'].value_counts().sort_index()
#vc_median_freq_index = value_counts.sum()/2
#edex = value_counts.sum()/2
#median_make = value_counts[value_counts.cumsum() >= vc_median_freq_index].index[0]
#bxBooksUserRating['User-Age'] = bxBooksUserRating['User-Age'].fillna(median_make)
#bxBooksUserRating['binned_age'] = bxBooksUserRating['User-Age'].apply(bin_age)

#bin_age_freq = bxBooksUserRating.groupby('User-Age').size().reset_index(name='binned_age')
#bin_ranges = [0, 18 ,24, 29, 39, 49, 64]
#plt.figure(figsize=(8,5))
#sns.histplot(data=bin_age_freq, x='User-Age', weights='binned_age', bins=bin_ranges)

#plt.savefig('median_age_histplot.png')
#plt.show()
#------

# Dataset countries.csv
dfCountries = pd.read_csv('location.csv')
dfCountries['User-Country'] = dfCountries['Country'].apply(lambda x: x.lower())
dfCountries = dfCountries[['User-Country', 'Latitude', 'Longitude']]
origin_lat, origin_lon = 0, 0
dfCountries['Distance'] = dfCountries.apply(lambda x: euclidean_distance(origin_lat, origin_lon, x['Latitude'], x['Longitude']), axis=1)

# Countries Preprocessing to match the countries.csv
nan_pattern = r'n/a"'
usa_pattern = r'usa"'
america_pattern = r'america"'
uk_pattern = r'united kingdom"'
us_pattern = r'us"'
iran_pattern = r'iran"'
for index, countries in bxBooksUserRating["User-Country"].items():
    if re.search(nan_pattern, str(countries)):
        bxBooksUserRating.at[index, "User-Country"] = np.nan
    elif re.search(usa_pattern, str(countries)):
        bxBooksUserRating.at[index, "User-Country"] = "united states"
    elif re.search(america_pattern, str(countries)):
        bxBooksUserRating.at[index, "User-Country"] = "united states"
    elif re.search(uk_pattern, str(countries)):
        bxBooksUserRating.at[index, "User-Country"] = "united kingdom"
    elif re.search(us_pattern, str(countries)):
        bxBooksUserRating.at[index, "User-Country"] = "united states"
    elif re.search(iran_pattern, str(countries)):
        bxBooksUserRating.at[index, "User-Country"] = "iran, islamic republic of"
nan_countries = bxBooksUserRating["User-Country"].isna()
bxBooksUserRating = bxBooksUserRating[~nan_countries]

# Erasing " after country names
country_pattern = r'\w+"'
for index, country in bxBooksUserRating["User-Country"].items():
    if re.search(country_pattern, str(country)):
        cleaned_country = re.sub(r'"', '', str(country).strip()).lower()
        bxBooksUserRating.at[index, "User-Country"] = cleaned_country

# Preprocess Book Titles
bxBooksUserRating['Processed-Title'] = bxBooksUserRating['Book-Title'].apply(preprocess_text)

# Filter out empty titles after preprocessing
bxBooksUserRating = bxBooksUserRating[bxBooksUserRating['Processed-Title'].str.strip() != '']

# Ignoring Book-Author, Book-Publisher, Year-Of-Publication, User-City, User-State, ISBN
bxBooksUserRating = bxBooksUserRating[['Book-Title', 'Book-Rating', 'binned_rating', 'User-ID', 'User-Country', 'User-Age']]

#----- Pie chart United States
#bxBooksUserRating['is_us'] = bxBooksUserRating['User-Country'] == "united states"
#us_cluster = bxBooksUserRating.groupby('is_us').size()
#print(us_cluster)
#plt.figure(figsize=(3, 3))
#plt.pie(us_cluster, labels=us_cluster.index, autopct='%1.1f%%', startangle=140)
#plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
#plt.title('Percentage of US Country')
# Save the plot as a PNG file
#plt.savefig('us_distance_pie_chart.png')

# Show the plot
#plt.show()
#-----

# Merging with the countries.csv and calculate the distance with 0, 0 as the centre point
bxMerge = pd.merge(bxBooksUserRating, dfCountries, on='User-Country', how='left')
bxMerge = bxMerge[['Book-Title','Book-Rating', 'binned_rating', 'User-ID', 'Distance', 'User-Age']]
bxMerge = bxMerge[~bxMerge['Distance'].isna()]

# Pre-process the Book-Title
bxMerge['Book-Title'] = bxMerge['Book-Title'].apply(preprocess_text)

# Initializing the vectorizer and the kmeans function
vectorizer = TfidfVectorizer()
kmeans = KMeans(n_clusters=6, random_state=42)

# Fit and transform the Book-Title to the vextorizer and cluster it with K-Means
tfidf_matrix_bxMerge = vectorizer.fit_transform(bxMerge["Book-Title"])
bxMerge['Title-Cluster'] = kmeans.fit_predict(tfidf_matrix_bxMerge)

#----- KMeans Elbow
#distortions = []
#k_range = range(1, 10)
#for k in k_range:
#    kmeans = KMeans(n_clusters=k)
#    kmeans.fit(tfidf_matrix_bxMerge)
#    distortions.append(kmeans.inertia_) # Question: What does kmeans.inertia_ return? 
    
#plt.plot(k_range, distortions, 'bx-')
#plt.title('The Elbow Method showing the optimal k')
#plt.xlabel('k')
#plt.ylabel('Distortion')
#plt.savefig('elbow_kmeans.png')
#plt.show()

# KNN to fit the missing age
nan_age = bxMerge['User-Age'].isna()
train = bxMerge[~nan_age]
test = bxMerge[nan_age]
X_COLS = ['Distance']
y_COL = 'User-Age'
X_train = train[X_COLS]
y_train = train[y_COL]
X_test = test[X_COLS]
y_test = test[y_COL]

# Create and fit knn with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Fit to the train dataset
knn.fit(X_train, y_train)

# Binning the old dataset age
bxBooksUserRating['binned_age'] = bxBooksUserRating['User-Age'].apply(lambda x: bin_age(x))

# Predict the age and then bin them to the merged dataframe
pred_age = knn.predict(X_test)
bxMerge.loc[nan_age, 'User-Age'] = pred_age
bxMerge['binned_age'] = bxMerge['User-Age'].apply(lambda x: bin_age(x))

#----- KNN Hist
#bin = [0, 18 ,24, 29, 39, 49, 64]
#plt.figure(figsize=(8,5))
#sns.histplot(data=bxMerge, x='User-Age', bins=bin)
#plt.savefig('knn_age_histplot.png')
#plt.show()
#-----

# Merging the new dataset
nbxBooks = pd.read_csv('BX-NewBooks.csv')
nbxRatings = pd.read_csv('BX-NewBooksRatings.csv')
nbxUsers = pd.read_csv('BX-NewBooksUsers.csv')
nbxBooksRating = pd.merge(nbxBooks, nbxRatings, on='ISBN', how='left')
new_rating = pd.merge(nbxBooksRating, nbxUsers, on='User-ID', how='left')

# Binning the rating of the new dataset
new_rating['binned_rating'] = new_rating['Book-Rating'].apply(lambda x: bin_rating(x))

# Cleaning the age with "
age_pattern = r'\d+"'
for index, age in new_rating["User-Age"].items():
    if re.search(age_pattern, str(age)):
        cleaned_age = re.sub(r'"', '', str(age).strip()).lower()
        new_rating.at[index, "User-Age"] = cleaned_age
        
# Age Outliers Removal
new_rating['Book-Rating'] = pd.to_numeric(new_rating['Book-Rating'], errors='coerce')
new_rating['User-Age'] = pd.to_numeric(new_rating['User-Age'], errors='coerce')
age_outliers = (new_rating["User-Age"] < 6) | (new_rating["User-Age"] > 100)
new_rating["User-Age"] = new_rating["User-Age"][~age_outliers]

# Countries Preprocessing to match the countries.csv
nan_pattern = r'n/a"'
usa_pattern = r'usa"'
america_pattern = r'america"'
uk_pattern = r'united kingdom"'
us_pattern = r'us"'
iran_pattern = r'iran"'
for index, countries in new_rating["User-Country"].items():
    if re.search(nan_pattern, str(countries)):
        new_rating.at[index, "User-Country"] = np.nan
    elif re.search(usa_pattern, str(countries)):
        new_rating.at[index, "User-Country"] = "united states"
    elif re.search(america_pattern, str(countries)):
        new_rating.at[index, "User-Country"] = "united states"
    elif re.search(uk_pattern, str(countries)):
        new_rating.at[index, "User-Country"] = "united kingdom"
    elif re.search(us_pattern, str(countries)):
        new_rating.at[index, "User-Country"] = "united states"
    elif re.search(iran_pattern, str(countries)):
        new_rating.at[index, "User-Country"] = "iran, islamic republic of"
nan_countries = new_rating["User-Country"].isna()
new_rating = new_rating[~nan_countries]

# Erasing " after country names
country_pattern = r'\w+"'
for index, country in new_rating["User-Country"].items():
    if re.search(country_pattern, str(country)):
        cleaned_country = re.sub(r'"', '', str(country).strip()).lower()
        new_rating.at[index, "User-Country"] = cleaned_country

# Ignoring the same features that are ignored in the old dataset
new_rating = new_rating[['Book-Title', 'Book-Rating', 'binned_rating', 'User-ID', 'User-Country', 'User-Age']]

# Merging with the countries.csv and calculate the distance using 0, 0 as centre point
newMerge = pd.merge(new_rating, dfCountries, on='User-Country', how='left')
newMerge = newMerge[['Book-Title', 'Book-Rating','binned_rating', 'User-ID', 'Distance', 'User-Age']]
newMerge = newMerge[~newMerge['Distance'].isna()]

# Preprocess the Book Title
newMerge['Book-Title'] = newMerge['Book-Title'].apply(preprocess_text)

# Fit and transform the "Book-Title" column in bxBooksUserRating DataFrame
tfidf_matrix_newMerge = vectorizer.fit_transform(newMerge["Book-Title"])
newMerge['Title-Cluster'] = kmeans.fit_predict(tfidf_matrix_newMerge)

# KNN to predict the missing age
nan_age = newMerge['User-Age'].isna()
train = newMerge[~nan_age]
test = newMerge[nan_age]
X_COLS = ['Distance']
y_COL = 'User-Age'
X_train = train[X_COLS]
y_train = train[y_COL]
X_test = test[X_COLS]
y_test = test[y_COL]

# Fit to the train dataset
knn.fit(X_train, y_train)

# Predict the age and then bin them
pred_age = knn.predict(X_test)
newMerge.loc[nan_age, 'User-Age'] = pred_age
newMerge['binned_age'] = newMerge['User-Age'].apply(lambda x: bin_age(x))

# Calculate the user mean ratings for every given title cluster
user_cluster_ratings = bxMerge.groupby(['User-ID', 'Title-Cluster'])['Book-Rating'].mean().reset_index()
user_cluster_ratings.columns = ['User-ID', 'Title-Cluster', 'Mean-Rating']
newMerge = newMerge.merge(user_cluster_ratings, on=['User-ID', 'Title-Cluster'], how='left')

# Calculate the user mean rating for all clusters
user_mean_ratings_bxMerge = bxMerge.groupby('User-ID')['Book-Rating'].mean().reset_index()
user_mean_ratings_bxMerge.columns = ['User-ID', 'Mean-Rating-bxMerge']
newMerge = newMerge.merge(user_mean_ratings_bxMerge, on='User-ID', how='left')

#----- User Mean Rating Histogram
#bin_ranges = [0, 4 ,7, 10]
#plt.figure(figsize=(8,5))
#sns.histplot(data=user_mean_ratings_bxMerge, x='Mean-Rating-bxMerge', bins=bin_ranges)
#plt.title('Binned Mean Rating For Each User')
#plt.savefig('user_mean_rating_histplot.png')
#plt.show()
#-----

# Filling the missing values with the mean rating of all clusters
newMerge['Predicted-Rating'] = newMerge['Mean-Rating'].fillna(newMerge['Mean-Rating-bxMerge'])
newMerge['predicted_binned_rating'] = newMerge['Predicted-Rating'].apply(bin_rating)

# Accuracy Calculation
accuracy = accuracy_score(newMerge['binned_rating'], newMerge['predicted_binned_rating'])
print(f"Accuracy: {accuracy}")
mae = mean_absolute_error(newMerge['binned_rating'], newMerge['predicted_binned_rating'])
print(f"Mean Absolute Error (MAE): {mae:.4f}")

#---- Histogram Comparison
#fig, (bp1, bp2) = plt.subplots(1, 2, sharey = True) 
#fig.suptitle('Binned Rating Comparison') 
#sns.histplot(newMerge['binned_rating'], ax=bp1, bins=[0, 1, 2, 3], kde=False)
#bp1.set_title('Actual Binned Rating')
#bp1.set_xlabel('Binned Rating')
#bp1.set_ylabel('Frequency')
#sns.histplot(newMerge['predicted_binned_rating'], ax=bp2, bins=[0, 1, 2, 3], kde=False)
#bp2.set_title('Predicted Binned Rating')
#bp2.set_xlabel('Binned Rating')
#bp2.set_ylabel('Frequency')
#plt.tight_layout()
#plt.savefig('binned_rating_comparison.png')
#plt.show()
#plt.close()
#-----

#----- Confusion Matrix
#cm = confusion_matrix(newMerge['binned_rating'], newMerge['predicted_binned_rating'])
# Create a confusion matrix display object
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
# Plot the confusion matrix
#fig, ax = plt.subplots(figsize=(10, 7))
#disp.plot(ax=ax, cmap='Blues', values_format='d')
# Set titles and labels
#plt.title('Confusion Matrix of Binned Ratings')
#plt.xlabel('Predicted Label')
#plt.ylabel('True Label')
# Save the plot
#plt.savefig('confusion_matrix_binned_rating.png')
#plt.show()
#-----

#----- Title Cluster Pie Chart
#plt.figure(figsize=(5, 5))
#cluster_sizes = newMerge.groupby('Title-Cluster').size()
#plt.pie(cluster_sizes, labels=cluster_sizes.index, autopct='%1.1f%%', startangle=140)
#plt.axis('equal')
#plt.title('Distribution of Title Clusters')
#plt.savefig('title_cluster_pie_chart.png')
#plt.show()
#print(cluster_sizes)
#-----