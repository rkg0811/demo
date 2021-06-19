import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify, render_template

book= pd.read_csv("Book-Ratings.csv",sep=';', error_bad_lines=False, encoding="latin-1",low_memory=False)
book_isbn=pd.read_csv("Books.csv",sep=';', error_bad_lines=False, encoding="latin-1",low_memory=False)

book_details=book_isbn[["ISBN","Book-Title","Book-Author","Year-Of-Publication","Publisher","Image-URL-M"]]
Merge_data = book.merge(book_details, on="ISBN", how = 'inner')

rating_total_count = pd.DataFrame(Merge_data.groupby('ISBN')['Book-Rating'].count())
rating_total_count.sort_values('Book-Rating', ascending=False)

user_total_count = pd.DataFrame(Merge_data.groupby('User-ID')['Book-Rating'].count())
user_total_count.sort_values('Book-Rating', ascending=False)

user_count = Merge_data['User-ID'].value_counts()
Merge_data = Merge_data[Merge_data['User-ID'].isin(user_count[user_count >= 100].index)]
book_count = Merge_data['Book-Rating'].value_counts()
Merge_data = Merge_data[Merge_data['Book-Rating'].isin(book_count[book_count >= 100].index)]
data=Merge_data[['User-ID','ISBN','Book-Rating','Book-Title','Book-Author','Year-Of-Publication','Publisher','Image-URL-M']]
data.rename(columns = {'User-ID':'userid','Book-Rating':'bookrating','Book-Title':'booktitle','Book-Author':'bookauthor', 'Year-Of-Publication':'publicationyear','Publisher':'publisher','Image-URL-M':'img'}, inplace = True)
data.reset_index()
data1 = data.merge(rating_total_count, left_on = 'ISBN', right_on = 'ISBN', how = 'left')
data1.rename(columns = {'Book-Rating':'totalratings'}, inplace = True)
threshold = 50
data1 = data1.query('totalratings >= @threshold')

data1 = data1.drop_duplicates(['userid', 'booktitle'])
data_pivot = data1.pivot(index = 'booktitle', columns = 'userid', values = 'bookrating').fillna(0)
data1_mat = csr_matrix(data_pivot.values)

# kNN Model

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(data1_mat)

pickle.dump(model_knn,open('model_train.pkl','wb'))

books_list=list(data_pivot.index)

with open("book_list.txt", "wb") as fp:
   pickle.dump(books_list, fp)

def get_index(req):
  count=0
  req=req.lower()
  for i in books_list:
    count+=1
    alpha=i.lower()
    
    if alpha==req:
      return (count-1)
    else:
      continue

app = Flask(__name__,template_folder='template')
model = pickle.load(open('model_train.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template("login.html")
database={'admin':'123','rishav':'rkg','gupta':'gpt','sample':'abc'}

@app.route('/form_login',methods=['POST','GET'])
def login():
    name1=request.form['username']
    pwd=request.form['password']
    if name1 not in database:
	    return render_template('login.html',info='Invalid User')
    else:
        if database[name1]!=pwd:
            return render_template('login.html',info='Invalid Password')
        else:
	         return render_template('book_index.html',name=name1)

@app.route('/predict',methods=['POST'])
def predict():
  if request.method == 'POST':
    message = str(request.form['message'])
    inp=str(message)
    ind=get_index(message)
    if bool(ind)== True:
      distances, indices = model.kneighbors(data_pivot.iloc[ind,:].values.reshape(1, -1), n_neighbors = 6)
      Listofbooks=[]
      for i in range(0, len(distances.flatten())):
        Listofbooks.append(data_pivot.index[indices.flatten()[i]])
      return render_template('book_index.html',prediction_text = 'Recommended Books for: {}'.format(Listofbooks[0]),
                             prediction_text1 = '1]  {}'.format(Listofbooks[1]),
                             prediction_text2 = '2]  {}'.format(Listofbooks[2]),
                             prediction_text3 = '3]  {}'.format(Listofbooks[3]),
                             prediction_text4 = '4]  {}'.format(Listofbooks[4]),
                             prediction_text5 = '5]  {}'.format(Listofbooks[5]))
    elif bool(ind)== False:
      return render_template('book_index.html',prediction_text = 'Sorry!! We Do not have that book in our Dataset')

if __name__ == "__main__":
  app.run()





