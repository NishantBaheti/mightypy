import torch
from torch import nn
from mightypy.datautils.download import FileDownloader
from mightypy.nlp.dataset import CustomDataset
from mightypy.nlp.llm import LLM, Word2Vec, train, generate
from tokkit import PyBytePairTokenizer, data_loader
from torch.utils.data import DataLoader
import yaml

N_HEADS = 10
D_MODEL = 256
D_KEY = 7
D_VALUE = 7
N_X = 10
CONTEXT_LENGTH = 100
DROPOUT = 0.4
VOCAB_SIZE = 500

EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

url = "https://raw.githubusercontent.com/NishantBaheti/tokkit/refs/heads/main/datasets/raw/combined.txt"
path = FileDownloader().save_file_local(url)

corpus = data_loader(path)
tokenizer = PyBytePairTokenizer()
tokenizer.fit(corpus, max_vocab_size=VOCAB_SIZE, n_iter=1e3)
VOCAB_SIZE = tokenizer.size
print("Vocab Size", VOCAB_SIZE)

dataset = CustomDataset(path, tokenizer, context_length=CONTEXT_LENGTH, device=device)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

embedding_model = Word2Vec(VOCAB_SIZE, D_MODEL, device=device)

llm_model = LLM(
    n_heads = N_HEADS,
    d_model = D_MODEL,
    d_key = D_KEY,
    d_value = D_VALUE,
    n_x = N_X,
    vocab_size = VOCAB_SIZE,
    dropout_p=DROPOUT,
    device=device
)

print(llm_model.total_params)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(llm_model.parameters(), lr=0.001, eps=1e-10, betas=(0.9, 0.98))

# out = model.forward(input_embeddings)
# print(out.shape)
# print(out.sum(dim=0))
with open('input_texts.yaml', 'r') as file:
    texts = yaml.safe_load(file)
for text in texts:
    print(text, " : " , generate(llm_model, embedding_model, tokenizer, text, 1000, 5, 1.0, device=device))

for i in range(EPOCHS):
    train(dataloader, llm_model, embedding_model, loss_fn, optimizer, 1, device)
    
    with open('input_texts.yaml', 'r') as file:
        texts = yaml.safe_load(file)
    for text in texts:
        print(text, " : " , generate(llm_model, embedding_model, tokenizer, text, 1000, 5, 1.0, device=device))


    if i % 10 == 0:
        torch.save(llm_model.state_dict(), f"models/llm_{datetime.now()}.pth")
        torch.save(embedding_model.state_dict(), f"models/embedding_{datetime.now()}.pth")