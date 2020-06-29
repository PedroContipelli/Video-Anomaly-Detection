interpolate_features = 0  # 0 - False, 1 - True
bags_per_video = 32
batch_size = 32

num_classes = 14
features = 'c3d'
input_dim = 4096
attention_layers = 1
num_heads = 8

num_epochs = 100
checkpoint = 5

alpha = 1e-4
beta = 1e-4

learning_rate = 1e-4
weight_decay = 1e-6
dropout_prob = 0.4
optimizer = 'ADAM'

train_percent = 1.0
validation_percent = 1.0

model_init = 'xavier_normal'

swa_start = 10
swa_update_interval = 3

