from imports import *

def tweet_clean(text, combined_pat, negations_dic,neg_pattern):
    clean1 = BeautifulSoup(text, 'lxml')
    clean2 = clean1.get_text()
    clean2.replace("ï¿½", "?")
    clean3 = re.sub(combined_pat, '', clean2)
    #remove upcases
    lower_case = clean3.lower()
    clean4 = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    #remove the non-letters
    clean5 = re.sub("[^a-zA-Z]", " ", clean4)
    words= [x for x in tok.tokenize(clean5) if len(x)>1]
    #remove the extra spaces
    clean = (" ".join(words)).strip()
    return clean
  
def plot_word_cloud(n, clean_df):
    #create a series with the most used words for positive/negative labels
    #n=1: positive tweets
    #n=0: negative tweets
    sent_tweets = clean_df[clean_df.sentiment == n]
    sent_string = []
    for t in sent_tweets.text:
        sent_string.append(t)
    sent_string = pd.Series(sent_string).str.cat(sep=' ')
    #show the most 100 used words
    wordcloud = WordCloud(width=2000, height=1000,colormap="viridis",max_words=100,background_color='white').generate(sent_string)
    plt.figure(figsize=(16,12))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    return(sent_string)
  
def plot_words(sent_string):
    words = {}
    for word in sent_string.split():
        if word not in words:
            words[word] = 1
        else:
            words[word] += 1
    word_counter = collections.Counter(words)
    common = word_counter.most_common(20)
    data_sent = pd.DataFrame(common, columns = ['Word', 'Count'])
    data_sent.plot.bar(x='Word',y='Count')
    return(words)
    
def accuracy_model(PL, x_train, y_train,x_validation, y_validation): #PL is the pipeline
    t0 = time()
    sentiment_fit = PL.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_validation)
    train_test_time = time() - t0
    model_accuracy = accuracy_score(y_validation, y_pred)
    print ("Accuracy score: "+ str(round(model_accuracy*100,2)))
    print ("Train and test time: "+ str(round(train_test_time,2))+" s")
    print ("--------"*10)
    return model_accuracy, train_test_time
  
def n_feature_tuner(vectorizer, n_features_list, stop_words, ngram_range, classifier, x_train, y_train,x_validation, y_validation):
    L = []
    classcopy=copy.copy(classifier)
    vectcopy=copy.copy(vectorizer)
    print (classcopy)
    print ("\n")
    for n in n_features_list:
        vectcopy.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        PL = Pipeline([('vectorizer', vectcopy),   ('classifier', classcopy)])
        print ("Validation result for "+ str(n)+ " " + " tokens:")
        nfeat_acc,time = accuracy_model(PL, x_train, y_train, x_validation, y_validation)
        L.append((n,nfeat_acc,time))
    return L

def model_tuner(vectorizer, n_features, stop_words, ngram_range, classifier,x_train, y_train):
    vectcopy=copy.copy(vectorizer)
    classcopy=copy.copy(classifier)
    vectcopy.set_params(stop_words=stop_words, max_features=n_features, ngram_range=ngram_range)
    PL = Pipeline([('vectorizer', vectcopy),   ('classifier', classcopy)])
    sentiment_fit =PL.fit(x_train, y_train)
    return sentiment_fit
  
def performance_report(model,x,y) :
    y_pred=model.predict(x).astype('int32')
    conf_matrix=pd.DataFrame(np.array(confusion_matrix(y, y_pred, labels=[0,1])), 
    index=['True Negatives', 'True Positives'],columns=['Predicted Negatives','Predicted Positives'])
    accuracy = accuracy_score(y, y_pred)
    target_names=["Negative","Positive"]
    performance_report=classification_report(y, y_pred, target_names=target_names,output_dict=True)
    print("\n")
    print(conf_matrix)
    print("\n")
    print(classification_report(y, y_pred, target_names=target_names))
    return(y_pred,conf_matrix,accuracy,performance_report)
    
