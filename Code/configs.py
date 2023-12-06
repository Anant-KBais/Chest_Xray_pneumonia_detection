from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator


class Allpaths():
    def __init__(self):
        self.dir_alldata = Path(r"C:\Users\anant\Desktop\Research\Data\chest_xray")
        self.train_data_dir = self.dir_alldata / 'train'
        self.validation_data_dir = self.dir_alldata / 'val'
        self.test_data_dir = self.dir_alldata / 'test'
        self.normal_cases_train = self.train_data_dir / 'NORMAL'
        self.pneumonia_cases_train = self.train_data_dir / 'PNEUMONIA'

        self.model = Path(r"C:\Users\anant\Desktop\Research\Code\model2")
paths = Allpaths()



def dataset_gen(paths, type, return_xy=False):

    batch_size = 20

    if type=='train' and return_xy ==False:
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        train_generator= train_datagen.flow_from_directory(paths.train_data_dir, target_size =(150,150),batch_size=batch_size, class_mode="binary" )
        return train_generator

    elif type=='train' and return_xy ==True:
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        train_generator= train_datagen.flow_from_directory(paths.train_data_dir, target_size =(150,150),batch_size=5217,shuffle=False, class_mode="binary" )
        x,y = train_generator.next()
        return x,y
    
    elif type == 'validation': 
        validation_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        validation_generator = validation_datagen.flow_from_directory(paths.validation_data_dir, target_size = (150,150),batch_size=batch_size, class_mode="binary")
        return validation_generator
    
    elif type == 'test' and return_xy ==False:
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(paths.test_data_dir, target_size = (150,150),shuffle=False,batch_size=batch_size, class_mode="binary")
        return test_generator
    
    elif type == 'test' and return_xy ==True:
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(paths.test_data_dir, target_size = (150,150),shuffle=False,batch_size=624, class_mode="binary")
        x, y = test_generator.next()
        return x, y
    


