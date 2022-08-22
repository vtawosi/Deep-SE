import numpy
import os
import sys
import pandas
from sklearn import metrics

# Author: Vali Tawosi
#
def main():
    # accepts two arguments, 1.input path, where the dataset files (separated into train and test sets) are
    # located (validation set has no use for the baseline methods), and 2. the dataset name
    args = sys.argv
    input_path = args[1]
    output_path = args[1]
    dataset = args[2]

    data_train = pandas.read_csv(input_path + dataset + '-train.csv').values
    data_test = pandas.read_csv(input_path + dataset + '-test.csv').values

    train_labels = data_train[:, 'storypoint'].astype('float32')
    test_labels = data_test[:, 'storypoint'].astype('float32')
    do_baselines('baselines', output_path, dataset, train_labels, test_labels)


def do_baselines(experiment, path, dataset, train_y, test_y):
    root_path = path + '/' + experiment
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if os.path.exists(root_path + '/' + dataset + '_rg_mae.csv'):
        rg_f = open(root_path + '/' + dataset + '_rg_mae.csv')
        mae_rguess = float(rg_f.read())
    else:
        mae_rguess = mae_random_guess(dataset, root_path, train_y, test_y)
        with open(root_path + '/' + dataset + '_rg_mae.csv', 'w') as f:
            f.write('%.4f' % mae_rguess)
            f.close()
    mean_mae, mean_mdae, mean_sa = mean_baseline(dataset, mae_rguess, root_path, test_y, train_y)
    median_mae, median_mdae, median_sa = median_baseline(dataset, mae_rguess, root_path, test_y, train_y)

    with open(root_path + '/' + dataset + '_baseline.csv', 'w') as f:
        f.write('mae_rguess, '
                'mean train, mean_mae, mean_mdae, mean_sa, '
                'median_train, median_mae, median_mdae, median_sa\n')
        f.write('%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f'
                % (mae_rguess,
                   numpy.mean(train_y), mean_mae, mean_mdae, mean_sa,
                   numpy.median(train_y), median_mae, median_mdae, median_sa))
        f.close()
    return mae_rguess


def mae_random_guess(dataset, path, sample, target):
    random_ae = None
    for i in range(1000):
        r_guess = numpy.random.choice(a=sample, size=len(target))
        if random_ae is None:
            random_ae = numpy.array(abs(target - r_guess))
        random_ae = numpy.vstack((random_ae, numpy.array(abs(target - r_guess))))
    mean_random_ae = numpy.mean(random_ae, axis=0)
    with open(path + '/' + dataset + '_random.txt', 'w') as random_f:
        for i in mean_random_ae:
            random_f.write('%.5f' % i)
            random_f.write('\n')
        random_f.close()
    return numpy.mean(mean_random_ae)  # MAE random guessing


def median_baseline(dataset, mae_rguess, path, test_y, train_y):
    median_list = [numpy.median(train_y)] * len(test_y)
    median_mae = metrics.mean_absolute_error(test_y, median_list)
    median_mdae = metrics.median_absolute_error(test_y, median_list)
    median_sa = (1 - (median_mae / mae_rguess)) * 100
    print 'median_train = %.5f' % numpy.median(train_y)
    print 'median_mae   = %.5f' % median_mae
    print 'median_mdae  = %.5f' % median_mdae
    print 'median_sa    = %.5f' % median_sa
    with open(path + '/' + dataset + '_median.txt', 'w') as median_f:
        ar = numpy.abs(numpy.subtract(test_y, median_list))
        for i in ar:
            median_f.write('%.5f' % i)
            median_f.write('\n')
        median_f.close()
    return median_mae, median_mdae, median_sa


def mean_baseline(dataset, mae_rguess, path, test_y, train_y):
    print('%s: mae_rguess = %f' % (dataset, mae_rguess))
    mean_list = [numpy.mean(train_y)] * len(test_y)
    mean_mae = metrics.mean_absolute_error(test_y, mean_list)
    mean_mdae = metrics.median_absolute_error(test_y, mean_list)
    mean_sa = (1 - (mean_mae / mae_rguess)) * 100
    print 'mean train = %.5f' % numpy.mean(train_y)
    print 'mean_mae   = %.5f' % mean_mae
    print 'mean_mdae  = %.5f' % mean_mdae
    print 'mean_sa    = %.5f' % mean_sa
    with open(path + '/' + dataset + '_mean.txt', 'w') as mean_f:
        ar = numpy.abs(numpy.subtract(test_y, mean_list))
        for i in ar:
            mean_f.write('%.5f' % i)
            mean_f.write('\n')
        mean_f.close()
    return mean_mae, mean_mdae, mean_sa


if __name__ == '__main__':
    main()
