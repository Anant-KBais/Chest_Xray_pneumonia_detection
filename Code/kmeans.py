
from keras.preprocessing.image import ImageDataGenerator
from configs import paths
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator= train_datagen.flow_from_directory(paths.train_data_dir, target_size =(150,150),batch_size=5216, class_mode="binary" )
 
test_generator = test_datagen.flow_from_directory(paths.test_data_dir, target_size = (150,150),batch_size=624, class_mode="binary")
from sklearn.metrics import classification_report

x, y = train_generator.next()
x_t, y_t = test_generator.next()
x1 = x.reshape(5216, 150*150*3)
x_t1 = x_t.reshape(624, 67500)
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(x1, y)
print(classification_report(y_t, clf.predict(x_t1))) 

# print(tree.plot_tree(clf, filled=True))