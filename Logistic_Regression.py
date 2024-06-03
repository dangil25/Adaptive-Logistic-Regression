import numpy as np
from math import log
#Currently set to classify images of digits
#WARNING: For 1024x1024 this takes ~10 minutes to run fully as it optimizes alpha and n_iter
#WARNING: Ignore overflow error if it occurs
#input: category + d dimensions
#d = dimension number, num = number of inputs, a = learning rate, n_iters = number of iterations

d = 64
num = 64


#train split
def train_test_split(X, y, train_size=0.25, random_state = 42):
    np.random.seed(random_state)
    X_train = np.random.choice(range(X.shape[0]), size=(int(len(X)*train_size),), replace=False)
    ind = np.zeros(X.shape[0], dtype=bool)
    ind[X_train] = True
    rest = ~ind
    X_train = X[ind]
    X_test = X[rest]

    y_train = y[ind]
    y_test = y[rest]

    return X_train, X_test, y_train, y_test

#separates input into categories, returns boolean array for each category showing if the specific element belongs to that category or not
def catsplit(y, cats):
    nY = []
    for i in range (len(cats)):
        nY.append([])
        for j in range (len(y)):
            if (y[j] == cats[i]): nY[i].append(1)
            else: nY[i].append(0)

    nY = np.array(nY)
    return nY

class LogisticRegression:
    def __init__(self, alpha, num_iters):
            self.alpha = alpha
            self.num_iters = num_iters
            self.w = None
            self.b = 0.2

    def sigmoid(self, z):
        return (1.0 / (1.0 + np.exp(-z)))

    #cost function for when we do predictions and check accuracy
    def cost(self, X, y):
        m, n = X.shape
        cost = 0.0
        for i in range (m):
            z_i = np.dot(X[i], self.w) + self.b
            calcprob = self.sigmoid(z_i)
            cost += -y[i]*log(calcprob) - (1 - y[i])*log(1 - calcprob)
        total_cost = cost/m
        return total_cost

    #calculates gradient for current weights
    def gradient(self, X, y):
        m, n = X.shape
        dw = np.zeros(self.w.shape)
        db = 0.0
        for i in range (m):
            z_i = np.dot(X[i], self.w) + self.b
            calcprob = self.sigmoid(z_i)
            error = calcprob - y[i]
            for j in range (n):
                dw[j] += error * X[i, j]
            db += error
        dw = dw/m
        db = db/m
        return dw, db

    #fits to data, moves against gradient and randomizes initial weights
    def fit(self, X, y):
        m, n = X.shape
        self.w = np.random.random(n) * 0.125 - 0.0625
        self.b = np.random.random() * 0.125 - 0.0625
        for i in range(self.num_iters):
            dw, db = self.gradient(X, y)
            self.w -= self.alpha * dw
            self.b -= self.alpha * db

    #returns probability for a certain inp being this cateogry
    def return_proba(self, inp):
        z = np.dot(inp, self.w) + self.b
        return self.sigmoid(z)

    #retutrns predictions for a full set
    def predict(self, X):
        m, n = X.shape
        y = np.zeros(m)
        for i in range (m):
            prob = self.return_proba(X[i])
            y[i] = prob
        return y

    #returns weights and intercept
    def returninfo(self):
        return self.w, self.b

#input stuff
Xp = []
yp = []
cats = []
for i in range (num):
    inp = input()
    inp = inp.split(',')
    for j in range (len(inp)):
        inp[j] = float(inp[j])
    Xp.append(inp[:-1])
    yp.append(inp[-1])
    if (yp[i] not in cats): cats.append(yp[i])

#sort cats for convenience
cats = sorted(cats)

#train test split stuff: X_train, y_train used to train, ~100 elements. Validate is used to optimize alpha and n_iter. final_test is for final statistics
X = np.array(Xp)
y = np.array(yp)
X_train, X_tests, y_train, y_tests = train_test_split(X, y, 0.1, 42)
X_validate, X_final_test, y_validate, y_final_test = train_test_split(X_tests, y_tests, 0.5, 42)

nY_Train = catsplit(y_train, cats)
nY_Test = catsplit(y_validate, cats)
nY_Final_Test = catsplit(y_final_test, cats)

#potential values for learning rate and n_iters
alphas = [4, 1, 0.25, 0.0625]
n_iters = [30, 90, 270]
maxcombo = [0, 0]
maxcorrect = 0.0
for alpha in alphas:
    for n_iter in n_iters:
        correct = 0
        probs = []
        regressions = [LogisticRegression(alpha, n_iter) for i in range (len(cats))]
        for i in range (len(cats)):
            regressions[i].fit(X_train, nY_Train[i])
            probs.append(regressions[i].predict(X_validate))
        for i in range(len(X_validate)):
            maxi = 0
            maxcat = 0
            for j in range(len(cats)):
                if (probs[j][i] > maxi):
                    maxi = probs[j][i]
                    maxcat = cats[j]
            if (maxcat == y_validate[i]):
                correct += 1
        correct /= len(X_validate)
        if (correct > maxcorrect):
            maxcorrect = correct
            maxcombo = [alpha, n_iter]

#recalculates the optimal regression
finalregressions = [LogisticRegression(maxcombo[0], maxcombo[1]) for i in range (len(cats))]
probs = []
correct = 0
for i in range(len(cats)):
    finalregressions[i].fit(X_train, nY_Train[i])
    probs.append(finalregressions[i].predict(X_final_test))
for i in range(len(X_final_test)):
    maxi = 0
    maxcat = 0
    for j in range(len(cats)):
        if (probs[j][i] > maxi):
            maxi = probs[j][i]
            maxcat = cats[j]
    if (maxcat == y_final_test[i]):
        correct += 1

#outputs to 6 decimal places
correct /= len(X_final_test)
print("Optimal alpha:", maxcombo[0], "Optimal n_iters:", maxcombo[1])
print("Final Test Results:")
print("% Correct:", round(correct * 100, 6), "\n")
for i in range (len(cats)):
    raw = finalregressions[i].returninfo()
    out = ""
    for num in raw[0]:
        out += str(round(num, 6)) + " "
    print(f"Weights for category {cats[i]}:", out)
    print(f"Intercept for category {cats[i]}:", str(round(raw[1], 6)), "\n")


