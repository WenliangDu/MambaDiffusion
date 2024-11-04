from .custom import CustomDataset
from .builder import DATASETS


@DATASETS.register_module()
class HunanDataset(CustomDataset):

    CLASSES = ('cropland', 'forest', 'grassland', 
               'wetland', 'unused land', 'water', 'built-up area',)

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], 
               [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30],]

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=True,
                 **kwargs):
        super(HunanDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label, **kwargs)