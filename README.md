<h1 align="center">🧠🧾 Article's Topics Classifier using NLP Pipeline</h1>

<h3 align="left"> End-to-End NLP pipeline for topics classification using deep learning ( LSTM ).</h3>
<p>In this Article I want to illustrate for the reader </p>
1. What is th NLP pipeline?
2. How to apply it's steps on a Rael-world project using python?

<h3>1. What is th NLP pipeline?</h3>
<p> It's a methodolgy facilitate for Data Scientists dealing with NLP projects as it divids the whole process into small tasks easy to deal with. </p>

1. Project Scope : You need to know what is your Idea?, how to apply it?, and how the user could get benefit from it?
2. Data Source : You can get your data from any source ( Database, API, Web Scrapping, ant etc).
3. Data Cleaning and Preprocessing : You have to make your data compatible for your needs in the NLP pipeline processes and Model training.
4. Model TrainingL You have two choices to apply (Classical ML or Deep Learning -prefered-)
5. Model Deployment: You need to make your project available for user as (Mobile app, Web app, Desktop app) service.
6. Push in production: After everything is done you need a server to make your Service available for user's action.



<h3>2. How to apply it's steps on a Rael-world project using python?</h3>
<p>Now we will discuss how to understand these steps and perform it using python Code</p>


<h5 align="left">Tools & Libraries 🛒:</h5>

- jupyter Notebook (python 3).
- BeautifulSoup for web scraping.
- NLTK for text processing.
- LSTM model - Logistic Regression & Naive Bayes.
- Flask for Deployment.

<p align="left"> In this project, my goal was to make a text classification by representing NLP-pipeline </p>

1. **Project Idea** 🎭: The project idea is to make a web service that is based on a deep learning model trained for text classification, the classification case is to take an Article Topic and classify this Topic into (ART, ECONOMY, SPORTS), this program can be very useful for many purposes in doing Analytics on user interests to improve websites reach and direct it for the convenient slots of users.



2. **Data Source** 📲: In my case, I've decided to use web scapping to create my own dataset which meets my needs, I've targeted a website of a famous Egyptian news newspaper, and I collected from it using Python code all the data needed to train the model so that each category had 10,000 texts to be trained on, with a total dataset of 30,000 rows.

<pre style=" border: 2px groove;
  border-radius: 5px;">
  <code style="color:tomato; line-height: 20%; ">
for i in range(20 , 10000, 20):
    src = requests.get("website_link"+str(i)+".extention").content
    soup= BeautifulSoup(src , "lxml")
    
    article_title=[]
    articles_title= soup.find_all("div",{"class":"col-md-12 col-lg-12 mar-top-outer"})
    
    for r in range(len(articles_title)):
        article_title.append(articles_title[r].text)        
    Articles_Scraper = pd.DataFrame({'Article Title': article_title,})

    Articles_Scraper['Article Title'].str.strip()
    Articles_Scraper['Category']='economy'
    print("iteration number",str(i),", Done")
    economy=pd.read_csv('economy.csv')
    economy = pd.concat([Articles_Scraper,economy])
    
    economy = economy.drop(columns=['Unnamed: 0'])
    economy.to_csv('economy_final.csv')
    print(economy.shape)
</code></pre>

3. **Data Cleaning** ✂: At the first of this process, we have 3 csv files each is about 10000 records.
- Concat them into one csv file.
- Drop duplicates if found.
- Remove spaces in the text.


<pre style=" border: 2px groove;
  border-radius: 5px;">
  <code style="color:tomato; line-height: 20%; ">df = pd.concat([art,economy,sports]).drop(columns=['Unnamed: 0'])
print(df.shape)
df.head()
df['Article Title']=df['Article Title'].str.strip()
df =df.reset_index(drop=True)
df = df[['Article Title','Category']]
df

</code></pre>

4. **Data Preprocessing** 🔧 : In this step, I have important processes to apply to my dataframe to be ready as an input for the model.
- Remove punctuation
- Convert all texts to be in lower case.
- Use nltk.tokenize for sentences tokenization.
- Remove stopwords from the tokenized text.
- Apply Stemming on the texts.
- Apply Lemmatization to the texts.
- Save the final processed dataframe to be used in the next step of Model Training.


<pre style=" border: 2px groove;
  border-radius: 5px;">
  <code style="color:tomato; line-height: 20%; ">
  
  #library that contains punctuation
import string
string.punctuation

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
#storing the puntuation free text
data['clean_msg']= data['Article Title'].apply(lambda x:remove_punctuation(x))

data['msg_lower']= data['clean_msg'].apply(lambda x: x.lower())

from nltk.tokenize import word_tokenize
data['tokenized_sents'] = data.apply(lambda row: word_tokenize(row['msg_lower']), axis=1)
data = data[['tokenized_sents','Category']]

import nltk
#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

#applying the function
data['no_stopwords']= data['tokenized_sents'].apply(lambda x:remove_stopwords(x))


#importing the Stemming function from nltk library
from nltk.stem.porter import PorterStemmer
#defining the object for stemming
porter_stemmer = PorterStemmer()

#defining a function for stemming
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text
data['msg_stemmed']=data['no_stopwords'].apply(lambda x: stemming(x))

data= data[['msg_stemmed','no_stopwords','Category']]

#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
data['msg_lemmatized']=data['no_stopwords'].apply(lambda x:lemmatizer(x))

data= data[['msg_stemmed','msg_lemmatized','Category']]
</code></pre>

5. **Model Training** 🏃‍♂️: The data was good enough to be fitted on the used models and the results was greatly satisfying specially whith deep learning model (LSTM) than classical ML using ( Logistic Regression & Naive Bayes ).


<pre style=" border: 2px groove;
  border-radius: 5px;">
  <code style="color:tomato; line-height: 20%; ">
    tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Final_Text'].values)

word_index = tokenizer.word_index
print("Unique Tokens are:",len(word_index))

X = tokenizer.texts_to_sequences(df['Final_Text'].values)
X = pad_sequences(X , maxlen=max_len)
print("Shape of data tensor",X.shape)
Y = pd.get_dummies(df['Category']).values
print("Shape of label tensor",Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.1, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = Sequential()
model.add(Embedding(n_most_common_words+1,emb_dim,input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy' , optimizer='adam' ,metrics=['accuracy'])
print(model.summary())

my_callbacks=[  EarlyStopping(monitor = 'val_loss',min_delta = 0,patience = 2,verbose = 1,restore_best_weights = True) ,
                ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5',monitor='val_loss',mode='min',save_best_only=True,verbose=1)  ]
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5,batch_size=128, callbacks=my_callbacks)
  
</code></pre>
6. **Project Deployment** 🎨: I used flask which is easy to use and give you the advantage of deploying your project as a web service.
<pre style=" border: 2px groove;
  border-radius: 5px;">
  <code style="color:tomato; line-height: 20%; ">
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['message']
    seq = tokenizer.texts_to_sequences([text])
    text = pad_sequences(seq, maxlen=max_len)
    classes = model.predict(text)
    labels=["Art","Economy","Sports"]
    prediction = str(labels[np.argmax(classes)])
    return render_template('After.html',data =prediction )

if __name__ == '__main__':
    app.run(debug=True)
  
</code></pre>

