import streamlit as st
import torch
import torch.nn as nn
import pickle
import re

# Load parameters
with open('params.pkl', 'rb') as f:
    params = pickle.load(f)

# Load vocabulary
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Define the model class
class SimpleRNNWithEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNNWithEmbeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# Instantiate and load the model
vocab_size = len(vocab)
model = SimpleRNNWithEmbeddings(
    vocab_size=vocab_size,
    embedding_dim=params['embedding_dim'],
    hidden_dim=params['hidden_dim'],
    output_dim=params['output_dim']
)

model.load_state_dict(torch.load('sentiment_model.pth', map_location=torch.device('cpu')))
model.eval()

# Preprocessing functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return tokens

def tokenize_and_numericalize(tokens, vocab):
    indices = [vocab.get(token, vocab.get('<unk>', 0)) for token in tokens]
    return indices

def pad_or_truncate(indices, max_length, padding_value=0):
    if len(indices) < max_length:
        indices.extend([padding_value] * (max_length - len(indices)))
    else:
        indices = indices[:max_length]
    return indices

# Streamlit app
def main():
    st.title("IMDB Sentiment Analysis")
    st.write("Enter a movie review to predict its sentiment.")

    user_input = st.text_area("Movie Review", "")

    if st.button("Predict"):
        if user_input.strip() != "":
            # Preprocess the input text
            tokens = preprocess_text(user_input)
            indices = tokenize_and_numericalize(tokens, vocab)
            indices = pad_or_truncate(indices, max_length=params['max_length'])

            # Convert to tensor and add batch dimension
            input_tensor = torch.tensor(indices).unsqueeze(0)  # Shape: (1, max_length)

            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][predicted_class].item()

            # Interpret prediction
            sentiment = "Positive" if predicted_class == 1 else "Negative"
            st.write(f"**Sentiment**: {sentiment}")
            st.write(f"**Confidence**: {confidence:.2f}")
        else:
            st.write("Please enter a review to analyze.")

if __name__ == '__main__':
    main()
