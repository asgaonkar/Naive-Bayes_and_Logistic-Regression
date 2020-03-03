import sys
import scipy.io
import numpy as np



#Load Data
#Do download the mnist_data.mat for label 7 and 8 (Handwritten Digits)
data = scipy.io.loadmat('mnist_data.mat') 

#Understanding Labels and Data
#7 = 0
#8 = 1
label_index = {"7":0, "8":1}

#Categorising into corresponding values
trX = data['trX']
trY = data['trY'] 
tsX = data['tsX'] 
tsY = data['tsY']

#Feauture Count
f_count = 2

#New Feature Sets
trX_Final = np.zeros(trX.shape[0]*f_count).reshape(trX.shape[0],f_count)
tsX_Final = np.zeros(tsX.shape[0]*f_count).reshape(tsX.shape[0],f_count)

shape = [trX.shape[0],f_count]

#Calculating Priors
sample_count = np.zeros(len(set(trY[0])))
for i in trY[0]:
    if(i==0):
        sample_count[0] += 1
    else:
        sample_count[1] += 1

sample_count_test_label = np.zeros(len(set(tsY[0])))
for i in tsY[0]:
    if(i==0):
        sample_count_test_label[0] += 1
    else:
        sample_count_test_label[1] += 1

priors = []
for i in range(len(sample_count)):
    priors.append(sample_count[i]/sum(sample_count))
print("Prior: ",priors)


#Understanding Guassian parameters
means = np.zeros(f_count*len(sample_count)).reshape((f_count,len(sample_count)))
variances = np.zeros(f_count*len(sample_count)).reshape((f_count,len(sample_count)))

#Feature Extraction
for i in range(trX.shape[0]):
        trX_Final[i][0] = np.mean(trX[i]) 
        trX_Final[i][1] = np.var(trX[i])
for i in range(tsX.shape[0]):        
        tsX_Final[i][0] = np.mean(tsX[i]) 
        tsX_Final[i][1] = np.var(tsX[i])

trX_7 = [trX_Final[i] for i in range(trX.shape[0]) if trY[0][i]==0]
trX_8 = [trX_Final[i] for i in range(trX.shape[0]) if trY[0][i]==1]

trX_7, trX_8 = np.array(trX_7), np.array(trX_8)

#Categorising into particular labels
trX_7_mean =np.array([x[0] for x in trX_7])
trX_8_mean =np.array([x[0] for x in trX_8])
trX_7_var = np.array([x[1] for x in trX_7])
trX_8_var = np.array([x[1] for x in trX_8])

trX_means = np.array([trX_7_mean, trX_8_mean])
trX_vars = np.array([trX_7_var, trX_8_var])

#Mean Matrix
means[0][0] = np.mean(trX_means[0])
means[0][1] = np.mean(trX_means[1])
means[1][0] = np.mean(trX_vars[0])
means[1][1] = np.mean(trX_vars[1])

means_T = means.transpose()
means_T = np.matrix(means_T)

print("Mean Matrix: ",means)

#Variance Matrix
variances[0][0] = np.var(trX_means[0])
variances[0][1] = np.var(trX_means[1])
variances[1][0] = np.var(trX_vars[0])
variances[1][1] = np.var(trX_vars[1])

variances_T = variances.transpose()
variances_T = np.matrix(variances_T)

print("Variance Matrix: ",variances)

#Covariance Matrix


print("-Covariance Matrix-")
#Covariance of 7
cov_7 = np.zeros(4).reshape((2,2))
cov_7[0][0] = variances[0][0]
cov_7[1][1] = variances[1][0]
cov_7 = np.matrix(cov_7)

print("Covariance Matrix (Digit 7): ",cov_7)

# #If dependent
# cov_7_dep = np.matrix(np.cov(trX_7_mean,trX_7_var))

#Covariance of 8
cov_8 = np.zeros(4).reshape((2,2))
cov_8[0][0] = variances[0][1]
cov_8[1][1] = variances[1][1]
cov_8 = np.matrix(cov_8)

print("Covariance Matrix (Digit 8): ",cov_8)

# #If dependent
# cov_8_dep = np.matrix(np.cov(trX_8_mean,trX_8_var))



#Naive Bayes
def naive_bayes(feature):

    #Calculate Prediction
    prob = [0,0]

    #P(X|Y=0) Likelihood
    likelihood_0 = ( 1/np.sqrt(((2*np.pi)**2)*(np.linalg.det(cov_7))) ) * ( np.exp( (-0.5)*np.dot( np.dot( (np.subtract(feature,means_T[0].reshape((2,1)))).transpose(), np.linalg.inv(cov_7) ) , (np.subtract(feature_arg,means_T[0].reshape((2,1))))) ) ) 
    #P(Y=0|X) Posterior
    prob[0] = likelihood_0*priors[0]

    #P(X|Y=1) Likelihood
    likelihood_1 = ( 1/np.sqrt(((2*np.pi)**2)*(np.linalg.det(cov_8))) ) * ( np.exp( (-0.5)*np.dot( np.dot( (np.subtract(feature,means_T[1].reshape((2,1)))).transpose(), np.linalg.inv(cov_8) ) , (np.subtract(feature_arg,means_T[1].reshape((2,1))))) ) ) 
    #P(Y=1|X) Posterior
    prob[1] = likelihood_1*priors[1]

    print("Posterior Value: ",prob)

    #Label the image
    if(prob[0]>prob[1]):
        return 0
    else:
        return 1


#Logistic Regression

#Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Log Likelihood
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    return np.sum( target*scores - np.log(1 + np.exp(scores)) )

#Logistic Regression
def logistic_regression(features, labels, iterations, learning_rate):
    
    features = np.hstack((np.ones((features.shape[0], 1)), features))
        
    weights = np.zeros(features.shape[1])
    # weights = np.array([-3.0, 300.0, -400.0])

    for i in range(iterations):
        predictions = sigmoid(np.dot(features, weights))

        # Update weights with gradient
        weights += np.dot(features.T, labels - predictions) * learning_rate
         
        
        # Print log-likelihood (at certain interval)
        if i % 500 == 0:
            print("----------------------------------")
            print("Iteration: ",i)
            print("Log Likelihood: ",log_likelihood(features, labels, weights))
            print("Weights: ",weights)
        
    return weights

#Test Labels
labels_gaussian = [0]*len(tsX_Final)
labels_logistic = [0]*len(tsX_Final)

#Call Test Data

#Gaussian
for i in range(len(tsX_Final)):
    feature_arg = [tsX_Final[i][0], tsX_Final[i][1]]
    feature_arg = np.array(feature_arg).reshape((2,1))
    print("----------------------------------")
    print("Labelling Image: ",i)
    labels_gaussian[i] = naive_bayes(feature_arg)

#Logistic

#Training
learning_rate = 0.001
iterations = 100000
weights = logistic_regression(trX_Final, trY[0],iterations,learning_rate)

#Testing
test_data = np.hstack((np.ones((len(tsX_Final), 1)),tsX_Final))
label_values = np.dot(test_data, weights)
labels_logistic = np.round(sigmoid(label_values))

#Accuracy Calculator
print("----------------------------------")
print("------------Accuracies------------")
print("----------------------------------")
print("Gaussian Accuracy (Total): ",(labels_gaussian == tsY[0]).sum().astype(float)*100/len(labels_gaussian),"%")

count = [0,0]

for i in range(len(tsY[0])):
    if(labels_gaussian[i]==tsY[0][i] and labels_gaussian[i]==0):
        count[0]+=1
    if(labels_gaussian[i]==tsY[0][i] and labels_gaussian[i]==1):
        count[1]+=1
print("Gaussian Accuracy Label 7: ",count[0]*100/sample_count_test_label[0],"%")
print("Gaussian Accuracy Label 8: ",count[1]*100/sample_count_test_label[1],"%")

print("\nLogistic Accuracy (Total): ",(labels_logistic == tsY[0]).sum().astype(float)*100/len(labels_logistic),"%")

count = [0,0]

for i in range(len(tsY[0])):
    if(labels_logistic[i]==tsY[0][i] and labels_logistic[i]==0):
        count[0]+=1
    if(labels_logistic[i]==tsY[0][i] and labels_logistic[i]==1):
        count[1]+=1

print("Logistic Accuracy Label 7: ",count[0]*100/sample_count_test_label[0],"%")
print("Logistic Accuracy Label 8: ",count[1]*100/sample_count_test_label[1],"%")