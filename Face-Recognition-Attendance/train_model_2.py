import os
import pickle
import numpy as np
import cv2
from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from imgaug import augmenters as iaa  # For data augmentation

dataset_path = 'C:\\Users\\ADMIN\\Desktop\\sumago\\Face_Recognition\\User_Images'

model = FaceNet()

X = []
y = []

augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),  
    iaa.Affine(rotate=(-20, 20)),  
    iaa.Multiply((0.8, 1.2)),  
])

for user_folder in os.listdir(dataset_path):
    user_path = os.path.join(dataset_path, user_folder)
    
    for image_file in os.listdir(user_path):
        img_path = os.path.join(user_path, image_file)
        img = cv2.imread(img_path)
        
        if img is not None:
            img = cv2.resize(img, (160, 160))
            img_aug = augmenter(images=[img])[0]
            embeddings = model.embeddings([img, img_aug])
            X.extend(embeddings)
            y.extend([user_folder] * len(embeddings))

X = np.array(X)
y = np.array(y)
print(np.unique(y))

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

svm_clf = SVC(probability=True)
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.001, 0.01, 0.1]
}
grid_search = GridSearchCV(svm_clf, param_grid, cv=3)
grid_search.fit(X_train, y_train)
best_svm = grid_search.best_estimator_

rf_clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=200)

rf_clf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)

svm_acc = accuracy_score(y_test, best_svm.predict(X_test))
rf_acc = accuracy_score(y_test, rf_clf.predict(X_test))
xgb_acc = accuracy_score(y_test, xgb_clf.predict(X_test))

print(f"Accuracy with tuned SVM: {svm_acc * 100:.2f}%")
print(f"Accuracy with RandomForest: {rf_acc * 100:.2f}%")
print(f"Accuracy with XGBoost: {xgb_acc * 100:.2f}%")

best_classifier = max([(svm_acc, best_svm), (rf_acc, rf_clf), (xgb_acc, xgb_clf)], key=lambda x: x[0])[1]

with open('face_classifier.pkl', 'wb') as f:
    pickle.dump(best_classifier, f)

print(f"Best classifier saved: Accuracy {max(svm_acc, rf_acc, xgb_acc) * 100:.2f}%")
