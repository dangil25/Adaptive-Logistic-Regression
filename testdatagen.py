from random import random
inf = 1e9
mini = inf
maxi = 0
file = open("testdata", "w")
out = ""
for j in range (1024):
    features = []
    wsum = 0
    for i in range(1024):
        features.append(random())
        wsum += features[i]
    car = wsum//3.5 - 141
    print(car)
    if (car < 0): car = 0
    if (car > 9): car = 9
    out += str(int(car)) + ","
    for i in features:
        out += str(i) + ","
    out = out[:-1] + "\n"
file.write(out)
file.close()