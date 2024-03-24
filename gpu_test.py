import torch

def main():
    print('we are in test mode.')
    print('torch version: ', torch.__version__)

    use_cuda = torch.cuda.is_available()
    print('cuda status: ', use_cuda)

    print('cudnn version: ', torch.backends.cudnn.version())
    print('number of cuda devices: ', torch.cuda.device_count())
    print('device name: ', torch.cuda.get_device_name())
    print('device memory: ', torch.cuda.get_device_properties(0).total_memory)
    print('device cpu count', torch.cuda.get_device_properties(0).multi_processor_count)


if __name__ == "__main__":
    main()