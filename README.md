# NLP
# This notebook is my practise book. I have written the below code, based on the teaching from Daniel Mbourke's course - "TensorFlow Developer Certificate in 2023 - 64 hours course"  https://zerotomastery.io/courses/learn-tensorflow/
	
What we're going to cover

	• Downloading a text dataset
		• !wget "https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip" 
		
	• Visualizing text data
		• Check for GPU
		• Helper functions - 
			§ unzip_data
			§ create_tensorboard_callback
			§ plot_loss_curves,
			§ compare_historys
		• Shuffling data - sample function - train_df.sample(frac=1, random_state=42)
		• itertuples() - used in for loop to give a row wise information in a set
		• train_test_split
		
	• Converting text into numbers using tokenization
		TextVectorization
text_vectorizer.adapt(train_sentences) # Fit the text vectorizer to the training text
	
	• Turning our tokenized text into an embedding
		• words_in_vocab = text_vectorizer.get_vocabulary() # Get the unique words in the vocabulary
		• Embedding
		
	• Modelling a text dataset
		• Model 0: Naive Bayes (baseline)
			from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
			model_0 = Pipeline([
                    ("tfidf", TfidfVectorizer()), # convert words to numbers using tfidf
                    ("clf", MultinomialNB()) # model the text
])
			GlobalAveragePooling1D
		• Model 1: Feed-forward neural network (dense model)
		• Model 2: LSTM model
		• Model 3: GRU model
		• Model 4: Bidirectional-LSTM model
		• Model 5: 1D Convolutional Neural Network
		• Model 6: TensorFlow Hub Pretrained Feature Extractor
		• Model 7: Same as model 6 with 10% of training data

		Each experiment will go through the following steps:
			• Construct the model
			• Train the model
			• Make predictions with the model -
			• Track prediction evaluation metrics for later comparison
			
			score function - baseline_score = model_0.score(val_sentences, val_labels)
			Predict funtion - baseline_preds = model_0.predict(val_sentences)
			
			# Compile model
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
			
			# Get a summary of the model
model_1.summary()
			
			# Fit the model
model_1_history = model_1.fit(train_sentences, # input sentences can be a list of strings due to text preprocessing layer built-in model
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR, 
                                                                     experiment_name="simple_dense_model")])
			
	• Comparing the performance of each our models
		
		Evaluation
		• Accuracy
		• Precision
		• Recall
		• F1-score

		# Check the results
model_1.evaluate(val_sentences, val_labels)
		
		# checking the weights
		embedding.weights
		
		# get_layer() and get_weights()
		embed_weights = model_1.get_layer("embedding_1").get_weights()[0]
print(embed_weights.shape)
		
		# Visualise here
		https://tensorboard.dev/
		
		# Make predictions (these come back in the form of probabilities)
model_1_pred_probs = model_1.predict(val_sentences)
		
		# Turn prediction probabilities into single-dimension tensor of floats
model_1_preds = tf.squeeze(tf.round(model_1_pred_probs))
	
	• Combining our models into an ensemble
		# model evaluation comparison
		# Visualising the comparisons
		
		Visualizing learned embeddings
		https://www.tensorflow.org/text/guide/word_embeddings#retrieve_the_trained_word_embeddings_and_save_them_to_disk
		
		Comparing the performance of each of our models
		
		Combining our models (model ensembling/stacking)
			a. Averaging - Take the output prediction probabilities of each model for each sample, combine them and then average them.
			b. Majority vote (mode) - Make class predictions with each of your models on all samples, the predicted class is the one in majority. For example, if three different models predict [1, 0, 1] respectively, the majority class is 1, therefore, that would be the predicted label.
			c. Model stacking - Take the outputs of each of your chosen models and use them as inputs to another model.
		
	• Saving and loading a trained model
		• The HDF5 format.
			• model_6.save("model_6.h5")
			• # Load model with custom Hub Layer (required with HDF5 format)
loaded_model_6 = tf.keras.models.load_model("model_6.h5", 
                                            custom_objects={"KerasLayer": hub.KerasLayer})
			• # How does our loaded model perform?
loaded_model_6.evaluate(val_sentences, val_labels)
			
		• The SavedModel format (default).
			• model_6.save("model_6_SavedModel_format")
		
	• Find the most wrong predictions
	• The speed/score tradeoff
		# Calculate the time of predictions
import time
![image](https://user-images.githubusercontent.com/123597753/233126314-b932ed12-b3cb-492d-82ba-14088dc59baa.png)
