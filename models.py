import torch.nn as nn

class DNAEncoder(nn.Module): 
    def __init__(self, input_dim=40, hidden_dim=1024, output_dim=768, dropout_rate=0.2):
        super(DNAEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class DNAEncoderWithClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes, input_dim):
        super().__init__()
        self.encoder = DNAEncoder(input_dim=input_dim) 
        self.classifier = nn.Linear(latent_dim, num_classes)  

    def forward(self, x):
        z = self.encoder(x)          
        logits = self.classifier(z)  
        return z, logits
    

class DNAEncoderSwint(nn.Module): #hidden layer added
    def __init__(self, input_dim=40, hidden_dim=1536, output_dim=1536, dropout_rate=0.2):
        super(DNAEncoderSwint, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class DNAEncoderWithClassifierSwint(nn.Module):
    def __init__(self, latent_dim, num_classes, input_dim):
        super().__init__()
        self.encoder = DNAEncoderSwint(input_dim=input_dim) 
        self.classifier = nn.Linear(latent_dim, num_classes)  

    def forward(self, x):
        z = self.encoder(x)          
        logits = self.classifier(z)  
        return z, logits
    
class DNAEncoderEva(nn.Module): #hidden layer added
    def __init__(self, input_dim=40, hidden_dim=1408, output_dim=1408, dropout_rate=0.2):
        super(DNAEncoderEva, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class DNAEncoderWithClassifierEva(nn.Module):
    def __init__(self, latent_dim, num_classes, input_dim):
        super().__init__()
        self.encoder = DNAEncoderEva(input_dim=input_dim) 
        self.classifier = nn.Linear(latent_dim, num_classes)  

    def forward(self, x):
        z = self.encoder(x)          
        logits = self.classifier(z)  
        return z, logits
