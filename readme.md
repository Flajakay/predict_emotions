Mixed results. But! The potential is there! 


Emotion Classification Application using RoBERTa

This application classifies text into six emotion categories:
- Sadness (0): Feelings of loss, disappointment, or despair
- Joy (1): Feelings of happiness, contentment, or excitement
- Love (2): Feelings of affection, fondness, or passion
- Anger (3): Feelings of frustration, irritation, or fury
- Fear (4): Feelings of anxiety, nervousness, or terror
- Surprise (5): Feelings of astonishment, shock, or amazement

The application is structured into three main modules:
1. Data Module: For data loading, preprocessing, and splitting
2. Model Module: For training, saving, and loading the classifier
3. Prediction Module: For making predictions on new text input

Install the required dependencies:

```bash
pip install torch transformers pandas scikit-learn numpy
pip install torch transformers pandas scikit-learn numpy
```

Install nvidia cudnn, nvidia cuda toolkit

Train a new model using a CSV file:

```bash
python emotion_classifier.py train --train_file your_training_data.csv --epochs 3 --model_dir ./emotion_model
```

Optional parameters:
- `--test_file`: Path to a test CSV file (if not provided, 20% of training data will be used)
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--max_length`: Maximum sequence length (default: 128)
- `--accumulation_steps`: Number of steps to accumulate gradients (default: 1)


Predict the emotion of a single text input:

```bash
python emotion_classifier.py predict --model_dir ./emotion_model --text "I'm so excited about my new job!"
```

Or predict emotions for a batch of texts in a CSV file:

```bash
python emotion_classifier.py predict --model_dir ./emotion_model --file test_texts.csv
```
