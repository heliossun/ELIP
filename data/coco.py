import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption
from pycocotools.coco import COCO

class coco_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        #url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'captions_train2017.json'

        #download_url(url,ann_root)
        
        self.coco = COCO(os.path.join(ann_root,filename))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.ids = list(self.coco.anns.keys())
        
           
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):    
        ann_id = self.ids[index]
        ann = self.coco.anns[ann_id]
        img_id=ann['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image_path = os.path.join(self.image_root,path)        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, index


class coco_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):        
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        '''        
        
        filename = 'captions_val2017_a3.json'

        #download_url(url,ann_root)
        
        #self.annotation = list(json.load(open(os.path.join(ann_root,filename),'r'))['annotations'])
        self.coco = COCO(os.path.join(ann_root,filename))
        # if ids provided by get_paths, use split-specific ids
        self.ids = list(self.coco.anns.keys())
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.coco_pair = {}
        for i in self.ids:
            img_id = self.coco.anns[i]['image_id']
            caption = self.coco.anns[i]['caption']
            path = self.coco.loadImgs(img_id)[0]['file_name']
            if img_id not in self.coco_pair.keys():
                self.coco_pair[img_id] = [path, []]
            self.coco_pair[img_id][1].append(caption)
        txt_id = 0
        for img_id, d in enumerate(self.coco_pair.values()):
            self.image.append(d[0])
            self.img2txt[img_id] = []
            #print(d)
            for t in d[1][:5]:
                self.text.append(pre_caption(t,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.coco_pair)
    
    def __getitem__(self, index):    
        path = self.image[index]
        image = Image.open(os.path.join(self.image_root, path)).convert('RGB')
        image = self.transform(image)
        
        return image, index
    
    
class flickr30k_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
        filenames = {'val':'flickr30k_val.json','test':'flickr30k_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index    