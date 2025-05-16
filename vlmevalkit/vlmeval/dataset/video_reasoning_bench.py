import huggingface_hub
from huggingface_hub import snapshot_download
from ..smp import *
from .video_concat_dataset import ConcatVideoDataset
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..utils import track_progress_rich
import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
from .utils.video_reasoning_bench import *


FAIL_MSG = 'Failed to obtain answer via API.'


class VideoReasoningBench(VideoBaseDataset):

    MD5 = 'fe2ccd8c0c95485760e0bfa0fac7216a'
    TYPE = 'Video-VQA'

    def __init__(self, dataset='VideoReasoningBench', nframe=0, fps=-1):
        self.type_data_list = {
            'hrd': ('hrd.json', './videos', '.mp4'),
            'cup': ('cup.json', './videos', '.mp4'),
            'grid': ('grid.json', './videos', '.mp4'),
            'file_sys': ('file_sys.json', './videos', '.mp4'),
            'card': ('card.json', './videos', '.mp4'),
            'chip': ('chip.json', './videos', '.mp4'),
        }
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['VideoReasoningBench']
    

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def save_json(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def prepare_dataset(self, dataset_name='VideoReasoningBench', repo_id='lyx97/reasoning_videos'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not osp.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False

            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(pth, item['video'])):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def read_parquet(pth):
                import pandas as pd
                for task_name in self.type_data_list.keys():
                    json_path = osp.join(pth, f'{task_name}.json')
                    if not osp.exists(json_path):
                        data = pd.read_parquet(osp.join(pth, task_name, 'test-00000-of-00001.parquet'))
                        data.to_json(json_path, orient='records', lines=False)
                        data = self.load_json(json_path)
                        data = {d["id"]: d for d in data}
                        self.save_json(data, json_path)

            def unzip_videos(pth):
                import zipfile
                if not osp.exists(osp.join(pth, 'videos')):
                    zip_file = osp.join(pth, 'videos.zip')
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(pth)

            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if osp.exists(data_file) and md5(data_file) == self.MD5:
                    return
                self.data_list = []
                for k, v in self.type_data_list.items():
                    with open(osp.join(pth, v[0]), 'r') as f:
                        json_data = json.load(f)
                    for idx, data in json_data.items():
                        data_item = {
                            'demo': k,
                            'video': data['video'],
                            'question': data['question'],
                            'answer': data['answer'],
                            'dim': data['dim'],
                            'visible_time': data['visible_time'],
                            'num_state': data['num_state'],
                            'num_operation': data['num_operation'],
                            'states': data['states'],
                            'coords': data['coords'] if 'coords' in data else None
                        }
                        self.data_list.append(data_item)

                data_df = pd.DataFrame(self.data_list)
                data_df = data_df.assign(index=range(len(data_df)))
                data_df.to_csv(data_file, sep='\t', index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            read_parquet(dataset_path)
            unzip_videos(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def save_video_frames(self, line):
        vid_path = osp.join(self.data_root, line['video'])
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(line['video'])
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(line['video'], len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths

    def save_video_into_images(self, line):
        frame_paths = self.save_video_frames(line)
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question, answer = line['question'], line['answer']
        message = []
        video_path = osp.join(self.data_root, line['video'])
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            img_frame_paths = self.save_video_into_images(line)
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))
        message.append(dict(type='text', value=question))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-1106', 'exact_matching', 'gpt-4o', 'gpt-4o-1120']
        judge_kwargs.update({
            "max_tokens": 512,
            "temperature": 0.
        })

        suffix = eval_file.split('.')[-1]
        score_file = eval_file.replace(f'.{suffix}', f'_{model}_score.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 8)

        if not osp.exists(score_file):
            data = load(eval_file)
            if model != 'exact_matching':
                model = build_judge(**judge_kwargs)
            else:
                model = None

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                _ = track_progress_rich(
                    evaluate_video_reasoning_bench,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
            ans = load(tmp_file)
            for idx, item in data.iterrows():
                data.loc[idx, 'score'] = ans[idx]['rating']
                data.loc[idx, 'final_ans_match'] = ans[idx]['final_ans_match']
            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        return rating