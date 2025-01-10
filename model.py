import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import math
import torch
from torch import nn, Tensor
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import time
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import argparse

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EEGNet(nn.Module):
    def __init__(self, F1=16, eegnet_kernel_size=32, D=2, eeg_chans=22, eegnet_separable_kernel_size=16,
                 eegnet_pooling_1=8, eegnet_pooling_2=4, dropout=0.5):
        super(EEGNet, self).__init__()
        F2 = F1*D
        self.dropout = nn.Dropout(dropout)
        self.block1 = nn.Conv2d(1, F1, (1, eegnet_kernel_size), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.block2 = nn.Conv2d(F1, F1*D, (eeg_chans, 1), padding='valid', bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.elu = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, eegnet_pooling_1))
        self.block3 = nn.Conv2d(F1*D, F2, (1, eegnet_separable_kernel_size), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F1*D)
        self.avg_pool2 = nn.AvgPool2d((1, eegnet_pooling_2))

    def forward(self, x):
        x = self.block1(x)
        if debug_mode_flag: print('Shape of x after block1 of EEGNet: ', x.shape)
        x = self.bn1(x)

        x = self.block2(x)
        if debug_mode_flag: print('Shape of x after block2 of EEGNet: ', x.shape)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout(x)
        if debug_mode_flag: print('Shape of x before block3 of EEGNet: ', x.shape)

        x = self.block3(x)
        if debug_mode_flag: print('Shape of x after block3 of EEGNet: ', x.shape)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout(x)
        if debug_mode_flag: print('Shape of x by the end of EEGNet: ', x.shape)
        return x

class EEGTransformerNet(nn.Module):
    def __init__(self, nb_classes, sequence_length, eeg_chans=22,
                 F1=16, D=2, eegnet_kernel_size=32, dropout_eegnet=0.3, eegnet_pooling_1=5, eegnet_pooling_2=5, 
                 MSA_num_heads = 8, flag_positional_encoding=True, transformer_dim_feedforward=2048, num_transformer_layers=6):
        super(EEGTransformerNet, self).__init__()
        """
        F1 = the number of temporal filters
        F2 = number of spatial filters
        """

        F2 = F1 * D
        self.sequence_length_transformer = sequence_length//eegnet_pooling_1//eegnet_pooling_2

        self.eegnet = EEGNet(eeg_chans=eeg_chans, F1=F1, eegnet_kernel_size=eegnet_kernel_size, D=D, 
                             eegnet_pooling_1=eegnet_pooling_1, eegnet_pooling_2=eegnet_pooling_2, dropout=dropout_eegnet)
        self.linear = nn.Linear(self.sequence_length_transformer, nb_classes)
        
        self.flag_positional_encoding = flag_positional_encoding
        self.pos_encoder = PositionalEncoding(self.sequence_length_transformer, dropout=0.3)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.sequence_length_transformer,
            nhead=MSA_num_heads,
            dim_feedforward=transformer_dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_transformer_layers
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.eegnet(x)
        x = torch.squeeze(x)


        x = x.permute(2, 0, 1)
        seq_len_transformer, batch_size_transformer, channels_transformer = x.shape
        x = torch.cat((torch.zeros((seq_len_transformer, batch_size_transformer, 1), 
                                   requires_grad=True).to(device), x), 2)
        x = x.permute(2, 1, 0) 
        if debug_mode_flag: print('Shape of x before Transformer: ', x.shape)
        if self.flag_positional_encoding:
            x = x * math.sqrt(self.sequence_length_transformer)
            x = self.pos_encoder(x) 
        if debug_mode_flag: print('Positional Encoding Done!')
        if debug_mode_flag: print('Shape of x after Transformer: ', x.shape)
        x = self.transformer_encoder(x)  
        x = x[0,:,:].reshape(batch_size_transformer, -1) 


        if debug_mode_flag: print('Shape of x before linear layer: ', x.shape)
        x = self.linear(x)
        return x


class NumpyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.classes = seizure_types 
        self.data = []
        for class_index, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            class_files = os.listdir(class_dir)
            for file_name in class_files:
                data = np.load(os.path.join(class_dir, file_name))
                n_samples = data.shape[0]
                for i in range(n_samples):
                    self.data.append((data[i], class_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx]
        return data, label

def train(model: nn.Module, optimizer, train_dataloader, aS=4):
    model.train()  
    running_loss = 0.0
    optimizer.zero_grad()
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.float(), labels.long()  
        inputs, labels = inputs.to(device), labels.to(device)  
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        if (i+1) % aS == 0 or (i+1) == len(train_dataloader):
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
    torch.cuda.empty_cache()
    return running_loss / len(train_dataloader)

def evaluate(model: nn.Module, val_dataloader):
    model.eval()  
    val_running_loss = 0.
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.float(), labels.long()  
            inputs, labels = inputs.to(device), labels.to(device)  

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    torch.cuda.empty_cache()
    return val_running_loss / len(val_dataloader), val_correct / val_total

def evaluate_test(model: nn.Module, test_dataloader):
    model.eval()  
    test_running_loss = 0.
    test_correct = 0
    test_total = 0
    total_predicted = []
    total_labels = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.float(), labels.long()  
            inputs, labels = inputs.to(device), labels.to(device)  

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            total_predicted += predicted.to('cpu').numpy().tolist()
            total_labels += labels.to('cpu').numpy().tolist()

    return test_running_loss / len(test_dataloader), test_correct / test_total, total_predicted, total_labels

def plot_train_val_loss(loss_train_array, loss_val_array, results_save_folder):

    plt.figure()
    plt.clf()
    plt.title("Training and Validation Loss")
    plt.plot(loss_val_array,label="val")
    plt.plot(loss_train_array,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(results_save_folder, 'Train_loss_curve.png'))

def define_initial_hyperparameters(eegnet_F1=32, eegnet_D=2, eegnet_kernel_size=32, MSA_num_heads=4):
    dropout = 0.3  
    model = EEGTransformerNet(nb_classes = nclasses, eeg_chans=num_eeg_channels, sequence_length=sequence_length,
                    F1=eegnet_F1, D=eegnet_D, eegnet_kernel_size=eegnet_kernel_size, 
                     MSA_num_heads = MSA_num_heads, dropout_eegnet=dropout).to(device)

    model_stats = summary(model, input_size=(batch_size, num_eeg_channels, sequence_length), verbose=0)
    print(model_stats)

    
    train_dir = os.path.join(data_root, 'train')  # Specify the path to root directory containing preprocessed training data
    train_dataset = NumpyDataset(train_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    val_dir = os.path.join(data_root, 'val')  # Specify the path to root directory containing preprocessed validation data
    val_dataset = NumpyDataset(val_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dir = os.path.join(data_root, 'dev')  # Specify the path to root directory containing preprocessed testing data
    test_dataset = NumpyDataset(test_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    lr = 0.0001 # learning rate
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    STEPLR_period = 10.0
    STEPLR_gamma = 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, STEPLR_period, gamma=STEPLR_gamma)

    epochs = 1000 # Number of epochs to be run
    best_model = None

    loss_train_array, loss_val_array = [], []

    starting_time = datetime.now()

    dt_string = starting_time.strftime("%Y-%m-%d-%H-%M")
    print("Starting date and time =", dt_string)

    root_path_keywords = 'results_EEGNet_Transformer_my_own_model_hyperparameter_tuning_ver2'

    if not os.path.exists(root_path_keywords):
        os.mkdir(root_path_keywords)
    results_save_folder = os.path.join(root_path_keywords, 'binary_experiment_' + dt_string + str(args.eegnet_kernel_size))
    if not os.path.exists(results_save_folder):
        os.mkdir(results_save_folder)

    model_save_path = os.path.join(results_save_folder, 'best_model.pt')

    model_stats = summary(model, input_size=(batch_size, num_eeg_channels, sequence_length), verbose=0)

    model_summary_save_path = os.path.join(results_save_folder, 'model_summary.txt')
        
    with open(model_summary_save_path, 'w') as txtFile:
        for value in str(model_stats):
            txtFile.write(str(value))

    # Define the early stopping parameters
    patience = 50
    min_delta = 0.01


    best_model = None
    best_score = float('inf')
    best_epoch = -1
    no_improvement_counter = 0
    val_loss, val_acc, y_pred, y_test = evaluate_test(model, val_dataloader)
    y_test = np.array(y_test).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    print('=' * 89)
    print(f'| End of training | test loss {val_loss:5.2f} | '
        f'test acc {val_acc:8.2f}')
    print('=' * 89)
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(model, optimizer, train_dataloader)
        val_loss, val_acc = evaluate(model, val_dataloader)
        loss_train_array.append(train_loss)
        loss_val_array.append(val_loss)
        val_loss, val_acc, y_pred, y_test = evaluate_test(model, val_dataloader)
        y_test = np.array(y_test).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)
        elapsed = time.time() - epoch_start_time
        cm = metrics.confusion_matrix(y_test, y_pred)
        tn_train, fp_train, fn_train, tp_train = cm.ravel()
        val_sensitivity = tp_train / (tp_train + fn_train)
        val_specificity = tn_train / (tn_train + fp_train)
        
        print('-' * 89)
       
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'| training loss {train_loss:5.3f} | '
            f'| \nConfusion Matrix {cm}| '
            f'| training sensitivity {val_sensitivity}|'
            f'| training specificity {val_specificity}|'
            f'valid loss {val_loss:5.3f} | valid acc {val_acc:8.3f} ')
        print('-' * 89)      
        scheduler.step()

        if best_score > val_loss + min_delta:
            best_score = val_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1  

        if no_improvement_counter >= patience:
            print("Early stopping at epoch:", epoch)
            torch.save(best_model.state_dict(), model_save_path)
            early_stop_epoch = epoch
            break
        
        if epoch == epochs:
            torch.save(best_model.state_dict(), model_save_path)
            torch.cuda.empty_cache()

    plot_train_val_loss(loss_train_array, loss_val_array, results_save_folder) 

   
    ### Save the confusion matrix
    test_loss, test_acc, y_pred, y_test = evaluate_test(best_model, test_dataloader)
    y_test = np.array(y_test).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    print(y_test.shape)
    print(y_pred.shape)
    if len(y_pred) < len(y_test):
        y_test = np.delete(y_test, range(len(y_pred),len(y_test)))
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | '
        f'test acc {test_acc:8.2f}')
    print('=' * 89)

    cm = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['Non-Seizure', 'Seizure'])
    cm_display.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(results_save_folder, 'confusion_matrix.png'))
  
    results_save_path = os.path.join(results_save_folder, 'output_results_' + str(segment_interval) + '_sec_segments.txt')
        
    with open(results_save_path, 'w') as txtFile:
        for value in metrics.classification_report(y_test, y_pred, digits=3):
            txtFile.write(str(value))

    hyperparameters_save_path = os.path.join(results_save_folder, 'hyperparameters.txt')

    with open(hyperparameters_save_path, 'w') as txtFile:
        txtFile.write('Segment interval: ' + str(segment_interval))
        txtFile.write('\n')
        txtFile.write('Training epochs: ' + str(epochs))
        txtFile.write('\n')
        txtFile.write('Best model epoch: ' + str(best_epoch))
        txtFile.write('\n')        
        txtFile.write('Learning rate: ' + str(lr))
        txtFile.write('\n')
        txtFile.write('Decaying period: ' + str(STEPLR_period))
        txtFile.write('\n')
        txtFile.write('Decaying gamma rate: ' + str(STEPLR_gamma))
        txtFile.write('\n')
        txtFile.write('Time taken to run: ' + str(datetime.now() - starting_time))
        txtFile.write('\n')
        txtFile.write('Training data root: ' + str(data_root))
        txtFile.write('\n')
        txtFile.write('eegnet_F1: ' + str(eegnet_F1))
        txtFile.write('\n')
        txtFile.write('eegnet_D: ' + str(eegnet_D))
        txtFile.write('\n')
        txtFile.write('eegnet_kernel_size: ' + str(eegnet_kernel_size))
        txtFile.write('\n')
        txtFile.write('MSA_num_heads: ' + str(MSA_num_heads))

if __name__ == "__main__":   
    ### Global parameters
    segment_interval = 4
    resampleFS = 250
    sequence_length = resampleFS * segment_interval
    num_eeg_channels= 22
    batch_size = 512
    nclasses = 2
    seizure_types = ['bckg', 'seizure']
    debug_mode_flag = False
    data_root = "/xxx/SeizureData/PreProcessedData/" + 'segment_interval_'+str(segment_interval)+'_sec' # Path to preprocessed data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)


    parser = argparse.ArgumentParser()
    parser.add_argument('--eegnet_kernel_size',type=int, required=True)
    parser.add_argument('--eegnet_f1',type=int, required=True)
    parser.add_argument('--eegnet_D',type=int, required=True)
    parser.add_argument('--num_heads',type=int, required=True)
    args = parser.parse_args()

    eegnet_kernel_size = args.eegnet_kernel_size
    eegnet_F1 = args.eegnet_f1
    eegnet_D = args.eegnet_D
    num_heads = args.num_heads
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0], dtype=torch.float).to(device)) # Adjust weights of 1.0 and 2.0 for bckg and seiz class as needed

    define_initial_hyperparameters(eegnet_F1=eegnet_F1, eegnet_D=eegnet_D, eegnet_kernel_size=eegnet_kernel_size, MSA_num_heads=num_heads)
