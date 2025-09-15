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