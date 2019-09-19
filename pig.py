

from sklearn.model_selection import train_test_split
from loadfiles import loaddata


X,y = loaddata("/home/pig/data")
X1,X2,Y1,Y2 = train_test_split(test_size=.33, random_state=1337)


