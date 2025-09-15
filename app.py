import os
import re
import argparse

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

    predict_parser = subparsers.add_parser('predict', help='Predict emotions for text')
    predict_parser.add_argument('--model_dir', required=True, help='Directory containing the trained model')
    predict_parser.add_argument('--text', help='Input text for prediction')
    predict_parser.add_argument('--file', help='Path to CSV file with texts')

    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model on test data')
    evaluate_parser.add_argument('--model_dir', required=True, help='Directory containing the trained model')
    evaluate_parser.add_argument('--test_file', required=True, help='Path to test CSV file')

    args = parser.parse_args()

    if args.command == 'train':

        data_module = DataModule()

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
            learning_rate=args.learning_rate
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