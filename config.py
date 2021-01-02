import warnings


class DefaultConfig(object):
    load_img_path = None  
    load_txt_path = None

    # data parameters
   
    training_size = 10000
    query_size = 2000
    database_size = 18015
    batch_size = 128

    img_dir = './dataset/data/mirflickr/'
    imgname_mat_dir = './dataset/data/FAll/mirflickr25k-fall.mat'
    img_mat_dir = './dataset/data/IAll/mirflickr25k-iall.mat'
    tag_mat_dir = './dataset/data/YAll/mirflickr25k-yall.mat'
    label_mat_dir = './dataset/data/LAll/mirflickr25k-lall.mat'

    # hyper-parameters
    max_epoch = 100
    alpha=1
    beta=1.4
    gamma = 0.1
    bit = 64  
    y_dim=1386
    label_dim=24
    lr = 10 ** (-1.5)  

    use_gpu = True
    valid = True


    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()
