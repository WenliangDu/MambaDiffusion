from .custom import CustomDataset
from .builder import DATASETS


@DATASETS.register_module()
class WhuDataset(CustomDataset):

    CLASSES = ('farmland', 'city', 'village', 
               'water', 'forest', 'road', 'others',)

    PALETTE = [[139, 69, 19], [255, 0, 0], [255, 255, 0], 
               [0, 0, 255], [0, 255, 0], [0, 255, 255], [205, 96, 144],]

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=True,
                 **kwargs):
        super(WhuDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label, **kwargs)