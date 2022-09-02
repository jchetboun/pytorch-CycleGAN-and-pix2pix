from data.base_dataset import BaseDataset, get_params, get_transform
from allegroai import DataView
from PIL import Image


class ClearmlAlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        assert (self.opt.load_size >= self.opt.crop_size)
        self.input_frame = opt.input_frame
        self.output_frame = opt.output_frame
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        dataview = DataView()
        dataview.add_query(dataset_name=opt.clearml_name, version_name=opt.clearml_version)
        dataview.prefetch_files()
        self.data = dataview.to_list()

    def __getitem__(self, index):
        A_path = self.data[index][self.input_frame].get_local_source(raise_on_error=True)
        B_path = self.data[index][self.output_frame].get_local_source(raise_on_error=True)
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        A = A_transform(A)
        B = B_transform(B)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.data)
