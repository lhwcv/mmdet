import os.path as osp
import xml.etree.ElementTree as ET

import mmcv

from .builder import DATASETS
from .xml_style import XMLDataset
from mmdet.core import eval_map, eval_recalls
import  numpy as np

@DATASETS.register_module()
class HollyWoodHeadDataset(XMLDataset):
    """Reader for the HoollyWood Head dataset in PASCAL VOC format.
    """
    CLASSES = ('head', )

    def __init__(self, **kwargs):
        super(HollyWoodHeadDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from WIDERFace XML style annotation file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = f'{img_id}.jpeg'
            xml_path = osp.join(self.img_prefix, 'Annotations_Quantize',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            folder = "JPEGImages"#root.find('folder').text
            data_infos.append(
                dict(
                    id=img_id,
                    filename=osp.join(folder, filename),
                    width=width,
                    height=height))

        return data_infos

    def evaluate(self,
                 results,
                 metric=['mAP'],
                 logger=None,
                 proposal_nums=(4,10,50),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mAP']

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}

        for metric in metrics:
            if metric == 'mAP':
                assert isinstance(iou_thr, float)
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                eval_results['mAP'] = mean_ap
        return eval_results
