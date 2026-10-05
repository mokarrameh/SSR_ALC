import torch.nn as nn
import torch.nn.functional as F

class Classifier2048(nn.Module):
    def __init__(self, embedding_dim=2048, num_classes=6, dropout_rate=0.3):
        super(Classifier2048, self).__init__()

        self.classifer = nn.Sequential(
            # nn.Linear(embedding_dim, 1024),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.Dropout(dropout_rate),

            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.Dropout(dropout_rate),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.classifer(x)  # [batch_size, 64] -> [batch_size, num_classes]
        return x

class Classifier1024(nn.Module):
    def __init__(self, embedding_dim=1024, num_classes=6, dropout_rate=0.3):
        super(Classifier1024, self).__init__()

        self.classifer = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.Dropout(dropout_rate),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.classifer(x)  # [batch_size, 64] -> [batch_size, num_classes]
        return x
    
class Classifier512(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=6, dropout_rate=0.3):
        super(Classifier512, self).__init__()

        self.classifer = nn.Sequential(
            # nn.Linear(embedding_dim, 256),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.Dropout(dropout_rate),

            # nn.Linear(embedding_dim, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.Dropout(dropout_rate),

            nn.Linear(embedding_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(64, num_classes)
        )


    def forward(self, x):

        x = self.classifer(x)  # [batch_size, 64] -> [batch_size, num_classes]
        return x

class Classifier256(nn.Module):
    def __init__(self, embedding_dim=256, num_classes=8, dropout_rate=0.2):
        super(Classifier256, self).__init__()

        self.classifier = nn.Sequential(
            # nn.Linear(embedding_dim, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.Dropout(dropout_rate),

            nn.Linear(embedding_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
    
class Classifier2048_kv(nn.Module):
    def __init__(self, embedding_dim=2048, num_classes=6, dropout_rate=0.3):
        super(Classifier2048_kv, self).__init__()

        self.classifer = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.classifer(x)  # [batch_size, 64] -> [batch_size, num_classes]
        return x

class Classifier1024_kv(nn.Module):
    def __init__(self, embedding_dim=1024, num_classes=6, dropout_rate=0.3):
        super(Classifier1024_kv, self).__init__()

        self.classifer = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.classifer(x)  # [batch_size, 64] -> [batch_size, num_classes]
        return x
    
class Classifier512_kv(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=6, dropout_rate=0.3):
        super(Classifier512_kv, self).__init__()

        self.classifer = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.classifer(x)  # [batch_size, 64] -> [batch_size, num_classes]
        return x

class Classifier256_kv(nn.Module):
    def __init__(self, embedding_dim=256, num_classes=8, dropout_rate=0.2):
        super(Classifier256_kv, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# --- SIMPLE CNN ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (256//4) * (256//4), 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x