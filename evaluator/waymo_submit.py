import os
import pickle
import argparse
import uuid

from tqdm import tqdm


LABEL_TO_TYPE = {'Vehicle': 1, 'Pedestrian':2, 'Cyclist':4}

class UUIDGeneration():
    def __init__(self):
        self.mapping = {}
    def get_uuid(self,seed):
        if seed not in self.mapping:
            self.mapping[seed] = uuid.uuid4().hex
        return self.mapping[seed]

uuid_gen = UUIDGeneration()

def _create_pd_detection(preds, gts, output_path, object_id=False):
    """Creates a prediction objects file."""
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2

    objects = metrics_pb2.Objects()
    for token, pred in tqdm(preds.items()):
        obj = gts[token]

        box3d = pred["boxes_lidar"]
        scores = pred["score"]
        labels = pred["name"]

        if len(box3d) == 0: continue
        if object_id:
            obj_ids = pred['obj_ids']

        box3d_len = box3d.shape[0]
        for i in range(box3d_len):
            bbox  = box3d[i]
            score = scores[i]
            label = labels[i]

            o = metrics_pb2.Object()
            o.context_name = obj['sequence_name']
            o.frame_timestamp_micros = int(obj['time_stamp'])

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = bbox[0]
            box.center_y = bbox[1]
            box.center_z = bbox[2]
            box.length = bbox[3]
            box.width = bbox[4]
            box.height = bbox[5]
            box.heading = bbox[6]
            o.object.box.CopyFrom(box)
            o.score = score
            # Use correct type.
            o.object.type = LABEL_TO_TYPE[label] 
            if object_id:
                o.object.id = uuid_gen.get_uuid(obj['sequence_name']+'-'+str(obj_ids[i]))
            objects.objects.append(o)

    # Write objects to a file.
    path = os.path.join(output_path, 'pred.bin')
    print("results saved to {}".format(path))
    f = open(path, 'wb')
    f.write(objects.SerializeToString())
    f.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Waymo Submit")
    parser.add_argument("--pred_path", type=str)
    parser.add_argument("--gt_path", type=str, )
    parser.add_argument("--output_path", type=str, default='./')
    parser.add_argument("--object_id", action='store_true')
    args = parser.parse_args()
    return args


def reorganize_infos(infos):
    new_info = {}

    for idx in range(len(infos)):
        info = infos[idx]
        token = info['sequence_name'] + str(info['sample_idx'])
        new_info[token] = info

    return new_info 

if __name__ == "__main__":
    args = parse_args()

    # load gt infos
    with open(args.gt_path, 'rb') as f:
        gt_infos = pickle.load(f)
    gt_infos = reorganize_infos(gt_infos)

    # load pred infos
    with open(args.path, 'rb') as f:
        pred_infos = pickle.load(f)
    pred_infos = reorganize_infos(pred_infos)      

    _create_pd_detection(pred_infos, gt_infos, args.output_path, object_id=args.object_id)
