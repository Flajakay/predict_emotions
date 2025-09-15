class DataModule:
    """Data loading, preprocessing and splitting module."""

    def __init__(self, max_length=128):
        """
        Initialize the data module.

        Args:
            max_length (int): Maximum sequence length for tokenization
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
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