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
        Initialize a new DistilBERT model.
        """
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', 
            num_labels=self.num_labels
        )
        self.model.to(self.device)

    def train_model(self, train_dataloader, validation_dataloader=None, epochs=4, learning_rate=2e-5):
        """
        Train the DistilBERT model.

        Args:
            train_dataloader (DataLoader): Training data loader
            validation_dataloader (DataLoader, optional): Validation data loader
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate

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

            for batch in train_dataloader:
                batch = [b.to(self.device) for b in batch]
                input_ids, attention_mask, labels = batch

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                running_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

                scheduler.step()

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

            self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)

            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise