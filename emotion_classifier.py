import os
import re
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

EMOTION_MAP = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

class DataModule:
    """Data loading, preprocessing and splitting module."""

    def __init__(self, max_length=128):
        """
        Initialize the data module.

        Args:
            max_length (int): Maximum sequence length for tokenization
        """
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = max_length

    def clean_text(self, text):
        """
        Basic text cleaning function.

        Args:
            text (str): Input text

        Returns:
            str: Cleaned text
        """
        if isinstance(text, str):

            text = text.lower()

            text = re.sub(r'[^\w\s]', '', text)

            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return ""

    def load_data(self, csv_file):
        """
        Load data from CSV file.

        Args:
            csv_file (str): Path to CSV file

        Returns:
            pd.DataFrame: Loaded data
        """
        print(f"Loading data from {csv_file}...")
        try:
            data = pd.read_csv(csv_file)
            required_columns = ['text', 'label']

            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"CSV file must contain columns: {required_columns}")

            data['text'] = data['text'].apply(self.clean_text)

            data['label'] = data['label'].astype(int)

            if not all(data['label'].between(0, 5)):
                raise ValueError("Labels must be integers between 0 and 5")

            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def prepare_data(self, texts, labels=None):
        """
        Prepare data for model training or inference.

        Args:
            texts (list): List of text strings
            labels (list, optional): List of labels

        Returns:
            dict or tuple: Prepared data for training or inference
        """

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        if labels is not None:
            labels = torch.tensor(labels)
            return input_ids, attention_mask, labels
        else:
            return input_ids, attention_mask

    def split_data(self, data, test_size=0.2, random_state=42):
        """
        Split data into training and test sets.

        Args:
            data (pd.DataFrame): DataFrame with 'text' and 'label' columns
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility

        Returns:
            tuple: (train_texts, test_texts, train_labels, test_labels)
        """
        train_data, test_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=data['label']
        )

        train_texts = train_data['text'].tolist()
        train_labels = train_data['label'].tolist()

        test_texts = test_data['text'].tolist()
        test_labels = test_data['label'].tolist()

        return train_texts, test_texts, train_labels, test_labels

    def create_data_loaders(self, input_ids, attention_mask, labels=None, batch_size=16):
        """
        Create PyTorch DataLoaders for training or inference.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs
            attention_mask (torch.Tensor): Attention mask
            labels (torch.Tensor, optional): Labels tensor
            batch_size (int): Batch size

        Returns:
            DataLoader: PyTorch DataLoader
        """
        if labels is not None:
            dataset = TensorDataset(input_ids, attention_mask, labels)
        else:
            dataset = TensorDataset(input_ids, attention_mask)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=labels is not None  
        )

class ModelModule:
    """Model training, saving and loading module."""

    def __init__(self, num_labels=6):
        """
        Initialize the model module.

        Args:
            num_labels (int): Number of emotion classes
        """
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = None

    def initialize_model(self):
        """
        Initialize a new RoBERTa model.
        """
        self.model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base', 
            num_labels=self.num_labels
        )
        self.model.to(self.device)

    def train_model(self, train_dataloader, validation_dataloader=None, epochs=4, learning_rate=2e-5, batch_size=16, accumulation_steps=1):
        """
        Train the RoBERTa model.

        Args:
            train_dataloader (DataLoader): Training data loader
            validation_dataloader (DataLoader, optional): Validation data loader
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            batch_size (int): Batch size for training
            accumulation_steps (int): Number of steps for gradient accumulation

        Returns:
            dict: Training metrics
        """
        if self.model is None:
            self.initialize_model()

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        total_steps = len(train_dataloader) * epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=total_steps
        )

        training_stats = []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print('-' * 40)

            self.model.train()
            running_loss = 0.0

            print("Training...")

            optimizer.zero_grad()

            for step, batch in enumerate(train_dataloader):
                batch = [b.to(self.device) for b in batch]
                input_ids, attention_mask, labels = batch

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss

                loss = loss / accumulation_steps
                running_loss += loss.item() * accumulation_steps

                loss.backward()

                if (step + 1) % accumulation_steps == 0:

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    optimizer.step()

                    scheduler.step()

                    optimizer.zero_grad()

            if len(train_dataloader) % accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss = running_loss / len(train_dataloader)
            print(f"Training loss: {epoch_loss:.4f}")

            if validation_dataloader:
                val_accuracy, val_loss = self.evaluate_model(validation_dataloader)
                print(f"Validation loss: {val_loss:.4f}")
                print(f"Validation accuracy: {val_accuracy:.4f}")

                epoch_stats = {
                    'epoch': epoch + 1,
                    'training_loss': epoch_loss,
                    'validation_loss': val_loss,
                    'validation_accuracy': val_accuracy
                }
            else:
                epoch_stats = {
                    'epoch': epoch + 1,
                    'training_loss': epoch_loss
                }

            training_stats.append(epoch_stats)

        print("\nTraining complete!")
        return training_stats

    def evaluate_model(self, dataloader):
        """
        Evaluate the model on a dataset.

        Args:
            dataloader (DataLoader): DataLoader with evaluation data

        Returns:
            tuple: (accuracy, loss)
        """
        if self.model is None:
            raise ValueError("Model has not been initialized or loaded")

        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in dataloader:
            batch = [b.to(self.device) for b in batch]
            input_ids, attention_mask, labels = batch

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            loss = outputs.loss
            logits = outputs.logits

            total_eval_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()
            total_eval_accuracy += accuracy

        avg_accuracy = total_eval_accuracy / len(dataloader)
        avg_loss = total_eval_loss / len(dataloader)

        return avg_accuracy, avg_loss

    def detailed_evaluation(self, dataloader):
        """
        Perform detailed evaluation and return classification report.

        Args:
            dataloader (DataLoader): DataLoader with evaluation data

        Returns:
            tuple: (classification_report, predictions, actual_labels)
        """
        if self.model is None:
            raise ValueError("Model has not been initialized or loaded")

        self.model.eval()
        predictions = []
        actual_labels = []

        for batch in dataloader:
            batch = [b.to(self.device) for b in batch]
            input_ids, attention_mask, labels = batch

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            batch_labels = labels.cpu().numpy()

            predictions.extend(batch_preds)
            actual_labels.extend(batch_labels)

        report = classification_report(
            actual_labels, 
            predictions, 
            target_names=[EMOTION_MAP[i] for i in range(self.num_labels)],
            digits=4
        )

        return report, predictions, actual_labels

    def save_model(self, output_dir):
        """
        Save the trained model.

        Args:
            output_dir (str): Directory to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Saving model to {output_dir}")

        self.model.save_pretrained(output_dir)

        config_info = {
            "num_labels": self.num_labels
        }

        import json
        with open(os.path.join(output_dir, "emotion_config.json"), 'w') as f:
            json.dump(config_info, f)

        print("Model saved successfully!")

    def load_model(self, model_dir):
        """
        Load a saved model.

        Args:
            model_dir (str): Directory containing the saved model
        """
        try:
            print(f"Loading model from {model_dir}")

            import json
            with open(os.path.join(model_dir, "emotion_config.json"), 'r') as f:
                config_info = json.load(f)

            self.num_labels = config_info.get("num_labels", 6)

            self.model = RobertaForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)

            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

class PredictionModule:
    """Prediction module for text emotion classification."""

    def __init__(self, model_module, data_module):
        """
        Initialize the prediction module.

        Args:
            model_module (ModelModule): Trained model module
            data_module (DataModule): Data processing module
        """
        self.model_module = model_module
        self.data_module = data_module

    def predict_emotion(self, text):
        """
        Predict emotion for a single text input.

        Args:
            text (str): Input text

        Returns:
            dict: Prediction results with emotion label and probability
        """

        cleaned_text = self.data_module.clean_text(text)

        input_ids, attention_mask = self.data_module.prepare_data([cleaned_text])

        input_ids = input_ids.to(self.model_module.device)
        attention_mask = attention_mask.to(self.model_module.device)

        self.model_module.model.eval()

        with torch.no_grad():
            outputs = self.model_module.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        prediction = torch.argmax(logits, dim=1).item()
        probability = probabilities[0][prediction].item()

        all_probs = probabilities[0].cpu().numpy()
        emotion_probs = {EMOTION_MAP[i]: float(all_probs[i]) for i in range(len(all_probs))}

        return {
            "emotion_id": prediction,
            "emotion": EMOTION_MAP[prediction],
            "probability": probability,
            "all_probabilities": emotion_probs
        }

    def predict_batch(self, texts):
        """
        Predict emotions for a batch of texts.

        Args:
            texts (list): List of input texts

        Returns:
            list: List of prediction results
        """

        cleaned_texts = [self.data_module.clean_text(text) for text in texts]

        input_ids, attention_mask = self.data_module.prepare_data(cleaned_texts)

        dataloader = self.data_module.create_data_loaders(
            input_ids, 
            attention_mask, 
            batch_size=8
        )

        self.model_module.model.eval()

        predictions = []

        for batch in dataloader:
            batch = [b.to(self.model_module.device) for b in batch]
            batch_input_ids, batch_attention_mask = batch

            with torch.no_grad():
                outputs = self.model_module.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )

            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)

            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            batch_probs = probabilities.cpu().numpy()

            for i, pred in enumerate(batch_predictions):
                predictions.append({
                    "emotion_id": int(pred),
                    "emotion": EMOTION_MAP[pred],
                    "probability": float(batch_probs[i][pred]),
                    "all_probabilities": {
                        EMOTION_MAP[j]: float(batch_probs[i][j]) 
                        for j in range(len(batch_probs[i]))
                    }
                })

        return predictions

def main():
    """Main function to run the emotion classification application."""
    parser = argparse.ArgumentParser(description='Emotion Classification Application')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--train_file', required=True, help='Path to training CSV file')
    train_parser.add_argument('--test_file', required=False, help='Path to test CSV file')
    train_parser.add_argument('--model_dir', default='emotion_model', help='Directory to save the model')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    train_parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    train_parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length for tokenization')
    train_parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of steps for gradient accumulation')

    predict_parser = subparsers.add_parser('predict', help='Predict emotions for text')
    predict_parser.add_argument('--model_dir', required=True, help='Directory containing the trained model')
    predict_parser.add_argument('--text', help='Input text for prediction')
    predict_parser.add_argument('--file', help='Path to CSV file with texts')

    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model on test data')
    evaluate_parser.add_argument('--model_dir', required=True, help='Directory containing the trained model')
    evaluate_parser.add_argument('--test_file', required=True, help='Path to test CSV file')

    args = parser.parse_args()

    if args.command == 'train':

        data_module = DataModule(max_length=args.max_length)

        train_data = data_module.load_data(args.train_file)

        if args.test_file:
            test_data = data_module.load_data(args.test_file)
            train_texts = train_data['text'].tolist()
            train_labels = train_data['label'].tolist()
            test_texts = test_data['text'].tolist()
            test_labels = test_data['label'].tolist()
        else:
            train_texts, test_texts, train_labels, test_labels = data_module.split_data(train_data)

        train_input_ids, train_attention_mask, train_labels_tensor = data_module.prepare_data(
            train_texts, train_labels
        )
        test_input_ids, test_attention_mask, test_labels_tensor = data_module.prepare_data(
            test_texts, test_labels
        )

        train_dataloader = data_module.create_data_loaders(
            train_input_ids, train_attention_mask, train_labels_tensor, batch_size=args.batch_size
        )
        test_dataloader = data_module.create_data_loaders(
            test_input_ids, test_attention_mask, test_labels_tensor, batch_size=args.batch_size
        )

        model_module = ModelModule()
        model_module.initialize_model()

        print("\nStarting model training...")
        training_stats = model_module.train_model(
            train_dataloader, 
            test_dataloader, 
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps
        )

        print("\nEvaluating model on test data...")
        report, _, _ = model_module.detailed_evaluation(test_dataloader)
        print("\nClassification Report:")
        print(report)

        model_module.save_model(args.model_dir)

    elif args.command == 'predict':

        data_module = DataModule()

        model_module = ModelModule()
        model_module.load_model(args.model_dir)

        prediction_module = PredictionModule(model_module, data_module)

        if args.text:

            result = prediction_module.predict_emotion(args.text)

            print("\nPrediction Results:")
            print(f"Input text: {args.text}")
            print(f"Predicted emotion: {result['emotion']} (ID: {result['emotion_id']})")
            print(f"Confidence: {result['probability']:.4f}")

            print("\nAll emotion probabilities:")
            for emotion, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion}: {prob:.4f}")

        elif args.file:

            data = pd.read_csv(args.file)
            texts = data['text'].tolist()

            results = prediction_module.predict_batch(texts)

            print(f"\nPredicted {len(results)} texts:")
            for i, (text, result) in enumerate(zip(texts[:10], results[:10])):  
                print(f"\n{i+1}. Input: {text[:50]}{'...' if len(text) > 50 else ''}")
                print(f"   Prediction: {result['emotion']} (Confidence: {result['probability']:.4f})")

            if len(texts) > 10:
                print(f"\n... and {len(texts) - 10} more texts")

            if 'label' in data.columns:
                labels = data['label'].tolist()
                correct = sum(1 for res, label in zip(results, labels) if res['emotion_id'] == label)
                accuracy = correct / len(labels)
                print(f"\nAccuracy on file: {accuracy:.4f} ({correct}/{len(labels)} correct)")
        else:
            print("Error: Either --text or --file must be provided")

    elif args.command == 'evaluate':

        data_module = DataModule()

        test_data = data_module.load_data(args.test_file)
        test_texts = test_data['text'].tolist()
        test_labels = test_data['label'].tolist()

        test_input_ids, test_attention_mask, test_labels_tensor = data_module.prepare_data(
            test_texts, test_labels
        )

        test_dataloader = data_module.create_data_loaders(
            test_input_ids, test_attention_mask, test_labels_tensor
        )

        model_module = ModelModule()
        model_module.load_model(args.model_dir)

        print("\nEvaluating model on test data...")
        report, predictions, actual_labels = model_module.detailed_evaluation(test_dataloader)

        print("\nClassification Report:")
        print(report)

        print("\nSample predictions:")
        for i in range(min(5, len(test_texts))):
            pred_emotion = EMOTION_MAP[predictions[i]]
            true_emotion = EMOTION_MAP[actual_labels[i]]
            correct = "✓" if predictions[i] == actual_labels[i] else "✗"

            print(f"\nText: {test_texts[i][:50]}{'...' if len(test_texts[i]) > 50 else ''}")
            print(f"Predicted: {pred_emotion}, Actual: {true_emotion} {correct}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()