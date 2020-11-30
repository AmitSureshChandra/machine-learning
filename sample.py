from sklearn import tree

# point in format of (x,y) in format [x,y]

X = [
    [10, -30],[30,-30],[-40,-40],[10,30],[30,30],[-30,30],[20,40],[-40,-20],[-30,30],[-50,20]
]

# corresponding cordinate in x-y plane

Y = [
    '4','4','3','1','1','2','1','3','2','2'
]

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

print(clf.predict([[-20,60],[-30,50],[50,60]]))