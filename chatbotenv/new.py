import random
import pickle
import json
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import os

# this 
import nltk
nltk.download('punkt')
Lemmatizer = WordNetLemmatizer()

# Fix the file path - go up one directory to find intents.json
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
# intents_path = os.path.join(parent_dir, 'intents.json')
intents_path = os.path.join(parent_dir, 'D:\program djangoetc\chatbot\intents.json')

with open(intents_path) as f:
    intents = json.load(f)

words=[]
classes=[]
documents=[]
ignore_letters=['?','.','!','@','#','$','%','^','&','*','(',')']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList= nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        
words=[Lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words=sorted(set(words))
classes=sorted(set(classes))

# Save files in the parent directory
pickle.dump(words,open(os.path.join(parent_dir, 'words.pkl'),'wb'))
pickle.dump(classes,open(os.path.join(parent_dir, 'classes.pkl'), 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag=[]
    word_patterns=document[0]
    word_patterns= [Lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row=list(output_empty)
    output_row[classes.index(document[1])]=1
    training.append([bag,output_row])

# Move these operations OUTSIDE the for loop
random.shuffle(training)

# Convert training data to numpy arrays properly
trainX = []
trainY = []
for bag, output_row in training:
    trainX.append(bag)
    trainY.append(output_row)

trainX = np.array(trainX)
trainY = np.array(trainY)

model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd= tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
hist= model.fit(np.array(trainX), np.array(trainY), epochs= 200, batch_size=5, verbose=1)
model.save(os.path.join(parent_dir, 'chatbot_model.h5'))
print("Executing the model")

# ============================================================================
# ACCURACY CALCULATION CODE
# ============================================================================
print("\n" + "="*60)
print("ACCURACY ANALYSIS")
print("="*60)

# Make predictions on training data
predictions = model.predict(trainX)
predicted_classes = []
true_classes = []

for i, prediction in enumerate(predictions):
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    predicted_classes.append(predicted_class)
    
    # Get true class from training data
    true_class_index = np.argmax(trainY[i])
    true_class = classes[true_class_index]
    true_classes.append(true_class)

# Calculate overall accuracy
correct_predictions = sum(1 for pred, true in zip(predicted_classes, true_classes) if pred == true)
total_predictions = len(predicted_classes)
overall_accuracy = correct_predictions / total_predictions

print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f"Correct Predictions: {correct_predictions}/{total_predictions}")

# Calculate per-class accuracy
print("\n" + "-"*40)
print("PER-CLASS ACCURACY")
print("-"*40)

class_accuracy = {}
for class_name in classes:
    class_correct = 0
    class_total = 0
    
    for pred, true in zip(predicted_classes, true_classes):
        if true == class_name:
            class_total += 1
            if pred == true:
                class_correct += 1
    
    accuracy = class_correct / class_total if class_total > 0 else 0
    class_accuracy[class_name] = accuracy
    print(f"{class_name:15}: {accuracy:.4f} ({accuracy*100:.2f}%) - {class_correct}/{class_total}")

# Calculate confusion matrix
print("\n" + "-"*40)
print("CONFUSION MATRIX")
print("-"*40)

# Initialize confusion matrix
confusion_matrix = {}
for true_class in classes:
    confusion_matrix[true_class] = {}
    for pred_class in classes:
        confusion_matrix[true_class][pred_class] = 0

# Fill confusion matrix
for pred, true in zip(predicted_classes, true_classes):
    confusion_matrix[true][pred] += 1

# Print confusion matrix
print("True\Predicted:", end="")
for pred_class in classes:
    print(f"{pred_class:10}", end="")
print()

for true_class in classes:
    print(f"{true_class:10}", end="")
    for pred_class in classes:
        print(f"{confusion_matrix[true_class][pred_class]:10}", end="")
    print()

# Calculate precision, recall, and F1-score
print("\n" + "-"*40)
print("DETAILED METRICS")
print("-"*40)

for class_name in classes:
    # True positives
    tp = confusion_matrix[class_name][class_name]
    
    # False positives (sum of column minus true positives)
    fp = sum(confusion_matrix[other_class][class_name] for other_class in classes if other_class != class_name)
    
    # False negatives (sum of row minus true positives)
    fn = sum(confusion_matrix[class_name][other_class] for other_class in classes if other_class != class_name)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{class_name:15}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1_score:.4f}")
    print()

# Test with sample inputs
print("\n" + "-"*40)
print("SAMPLE PREDICTIONS")
print("-"*40)

test_inputs = [
    "Hi there",
    "How are you", 
    "Bye",
    "Thank you",
    "What is Simplilearn?",
    "Hello",
    "See you later",
    "Thanks for helping me"
]

for test_input in test_inputs:
    # Prepare input
    wordList = nltk.word_tokenize(test_input)
    bag = []
    word_patterns = [Lemmatizer.lemmatize(word.lower()) for word in wordList]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Make prediction
    prediction = model.predict(np.array([bag]), verbose=0)
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    confidence = np.max(prediction)
    
    print(f"Input: '{test_input:20}' -> Predicted: {predicted_class:10} (Confidence: {confidence:.3f})")

print("\n" + "="*60)
print("ACCURACY ANALYSIS COMPLETE")
print("="*60)


