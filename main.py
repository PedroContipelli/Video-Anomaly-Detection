import argparse
from datetime import datetime
from train import *
import parameters as params
from tensorboardX import SummaryWriter


def train_classifier(run_id, use_cuda, args):
    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writer = SummaryWriter(os.path.join(cfg.tf_logs_dir, str(run_id)))
    for arg in vars(args):
        writer.add_text(arg, str(getattr(args, arg)))
    train_model(run_id, save_dir, use_cuda, args, writer)


def main(args):
    print("Run description : ", args.run_description)

    # call a function depending on the 'mode' parameter
    if args.train_classifier:
        run_id = args.run_id + '_' + datetime.today().strftime('%d-%m-%y_%H%M')
        use_cuda = torch.cuda.is_available()
        train_classifier(run_id, use_cuda, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train Anomaly Classifier model')

    # 'mode' parameter (mutually exclusive group) with five modes : train/test classifier, train/test generator, test
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--train_classifier', dest='train_classifier', action='store_true',
                       help='Training the Classifier for Anomaly Detection')

    parser.add_argument("--gpu", dest='gpu', type=str, required=False,
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument('--run_id', dest='run_id', type=str, required=False,
                        help='Please provide an ID for the current run')
    parser.add_argument('--run_description', dest='run_description', type=str, required=False,
                        help='Please description of the run to write to log')

    parser.add_argument('--features', type=str, default=params.features, help='Features to use.',
                        choices=["c3d", "i3d", "r2p1d"])
    parser.add_argument('--interpolate_features', type=int, default=params.interpolate_features,
                        help='Flag to interpolate features.')
    parser.add_argument('--bags_per_video', type=int, default=params.bags_per_video,
                        help='Flag to interpolate features.')

    parser.add_argument('--batch_size', type=int, default=params.batch_size, help='Batch size.')

    parser.add_argument('--num_classes', type=int, default=params.num_classes,
                        help='Number of anomaly classes.')
    parser.add_argument('--input_dim', type=int, default=params.input_dim,
                        help='Input feature dimension.')
    parser.add_argument('--attention_layers', type=int, default=params.attention_layers, help='Layers of Attention')
    parser.add_argument('--num_heads', type=int, default=params.num_heads, help='Number of heads')

    parser.add_argument('--num_epochs', type=int, default=params.num_epochs,
                        help='Number of epochs.')

    parser.add_argument('--learning_rate', type=float, default=params.learning_rate,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=params.weight_decay,
                        help='Weight decay.')
    parser.add_argument('--dropout', type=float, default=params.dropout_prob,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--optimizer', type=str, default=params.optimizer,
                        help='provide optimizer preference')
    parser.add_argument('--model_name', type=str, required=False,
                        help='provide model preference for training')
    parser.add_argument('--model_init', type=str, default=params.model_init,
                        help="Model weight initialization.")

    parser.add_argument('--swa_start', type=int, default=params.swa_start,
                        help='start epoch for SWA')
    parser.add_argument('--swa_update_interval', type=int, default=params.swa_update_interval,
                        help="Update interval for SWA")

    # parse arguments
    args = parser.parse_args()

    # set environment variables to use GPU-0 by default
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # exit when the mode is 'train_action_classifier' and the parameter 'run_id' is missing
    if args.train_classifier:
        if args.run_id is None:
            parser.print_help()
            exit(1)

    main(args)
