import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn, einsum, optim
from einops import rearrange, repeat
from torch.utils.data import DataLoader, Dataset
# feedforward and attention

class ContinuousDataset(Dataset):
    def __init__(self, X_cont, y):
        self.X_cont = X_cont
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'x_cont': torch.tensor(self.X_cont[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx], dtype=torch.float32)
        }

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# numerical embedder

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# main class

class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn = False):
        #assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim = 1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        # attend

        x, attns = self.transformer(x, return_attn = True)

        # get cls token

        x = x[:, 0]

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        if not return_attn:
            return logits

        return logits, attns


class FTTransformerWrapper:
    def __init__(self, transformer_model, lr=0.001, batch_size=64):
        """
        Initialize the FTTransformer model and define the optimizer and loss function.
        Args:
        - transformer_model: The FTTransformer model instance.
        - lr: Learning rate for the optimizer.
        - batch_size: Batch size for training.
        """
        self.transformer_model = transformer_model  # Direct reference to the model
        self.optimizer = optim.Adam(self.transformer_model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size

    def fit(self, X, y, epochs=50):
        """
        Train the FTTransformer model on the provided data.
        Args:
        - X: Continuous feature matrix (numpy array or pandas DataFrame).
        - y: Target labels (numpy array or pandas Series).
        - epochs: Number of epochs to train.
        """
        dataset = ContinuousDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.transformer_model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for batch in train_loader:
                self.optimizer.zero_grad()

                x_cont = batch['x_cont']
                y_true = batch['y']

                # Forward pass
                y_pred = self.transformer_model([], x_cont).squeeze()  # Directly use transformer_model

                # Compute loss
                loss = self.criterion(y_pred, y_true)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * x_cont.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            #print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

    def predict(self, X):
        """
        Make predictions using the trained model on the test dataset.
        Args:
        - X: Continuous feature matrix (numpy array or pandas DataFrame).
        Returns:
        - predictions: List of predicted labels (rounded).
        """
        self.transformer_model.eval()
        dataset = ContinuousDataset(X, torch.zeros(X.shape[0]))  # Dummy target
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                x_cont = batch['x_cont']
                y_pred = self.transformer_model([],x_cont).squeeze()  # Use transformer_model

                predictions.extend(torch.sigmoid(y_pred).round().cpu().numpy())

        return predictions

    def evaluate(self, X, y):
        """
        Evaluate the model on the test data and print accuracy.
        Args:
        - X: Continuous feature matrix (numpy array or pandas DataFrame).
        - y: True labels for the test data.
        """
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        #print(f'Test Accuracy: {accuracy:.4f}')
        return accuracy