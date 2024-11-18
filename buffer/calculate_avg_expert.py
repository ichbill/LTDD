import re
import argparse
import statistics

def process_log_file(file_path, total_epochs):
    last_epoch_acc = []
    epoch_regex = re.compile(r"Epoch: (\d+)")
    test_acc_regex = re.compile(r"Test Acc: ([0-9.]+)")

    last_epoch = total_epochs - 1

    with open(file_path, 'r') as file:
        for line in file:
            if f"Epoch: {last_epoch}" in line:
                epoch_match = epoch_regex.search(line)
                test_acc_match = test_acc_regex.search(line)
                if epoch_match and test_acc_match:
                    epoch = int(epoch_match.group(1))
                    test_acc = float(test_acc_match.group(1))
                    if epoch == last_epoch:
                        last_epoch_acc.append(test_acc)

    if last_epoch_acc:
        average_acc = sum(last_epoch_acc) / len(last_epoch_acc)
        std_acc = statistics.stdev(last_epoch_acc)
        return average_acc, std_acc
    else:
        return None

def main(args):
    # Use the function
    average_test_acc, std_acc = process_log_file(args.log_file_path, args.last_epoch)
    if average_test_acc is not None:
        print(f"Average Test Accuracy at Epoch {args.last_epoch} across all iterations: {average_test_acc*100}")
        print(f"Standard Deviation of Test Accuracy at Epoch {args.last_epoch} across all iterations: {std_acc*100}")
    else:
        print("No data found for Epoch 99.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--log_file_path', type=str, default='path_to_your_log_file.txt', help='path to the log file')
    parser.add_argument('--last_epoch', type=int, default=99, help='last epoch to calculate average test accuracy')
    args = parser.parse_args()
    main(args)


