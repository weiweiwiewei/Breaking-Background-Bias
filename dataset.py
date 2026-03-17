
import os
import json
import hashlib
from torch.utils.data import Dataset
from PIL import Image


class ImageNetAnimalFusionDataset(Dataset):
    
    def __init__(self, imagenet_root, class_list_file, split='train', transform=None, use_cache=True, logger=None):
        self.imagenet_root = imagenet_root
        self.class_list_file = class_list_file
        self.split = split
        self.transform = transform
        self.use_cache = use_cache
        self.logger = logger
        self.samples = []
        self.class_names = []
        self.class_to_idx = {}
        
        self._load_class_list(class_list_file)
        
        self._load_data()
    
    def _load_class_list(self, class_list_file):
        self.class_ids = []
        self.class_id_to_name = {}
        
        if class_list_file and os.path.exists(class_list_file):
            with open(class_list_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            class_id = parts[0].strip()
                            class_name = parts[1].strip()
                            self.class_ids.append(class_id)
                            self.class_id_to_name[class_id] = class_name
        
        split_dir = os.path.join(self.imagenet_root, self.split)
        if os.path.exists(split_dir):
            actual_classes = []
            if self.split == 'unknow1':
                vlcs_domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007', 'NICO']
                for domain in vlcs_domains:
                    domain_dir = os.path.join(split_dir, domain)
                    if os.path.exists(domain_dir):
                        for item in os.listdir(domain_dir):
                            item_path = os.path.join(domain_dir, item)
                            if os.path.isdir(item_path) and item not in actual_classes:
                                actual_classes.append(item)
            else:
                for item in os.listdir(split_dir):
                    item_path = os.path.join(split_dir, item)
                    if os.path.isdir(item_path):
                        actual_classes.append(item)
            
            if len(actual_classes) > 0:
                is_class_id_format = any(c.startswith('n') and len(c) > 1 and c[1:].isdigit() for c in actual_classes[:5])
                
                if is_class_id_format:
                    if len(self.class_ids) == 0:
                        self.class_ids = sorted(actual_classes)
                        for class_id in self.class_ids:
                            self.class_id_to_name[class_id] = class_id  # 如果没有名称，使用ID作为名称
                        if self.logger is not None:
                            self.logger.info(f"从数据目录自动发现 {len(self.class_ids)} 个类别（ImageNet格式）")
                else:
                    if len(self.class_ids) == 0:
                        self.class_ids = sorted(actual_classes)
                        for class_name in self.class_ids:
                            self.class_id_to_name[class_name] = class_name
                        if self.logger is not None:
                            self.logger.info(f"从数据目录自动发现 {len(self.class_ids)} 个类别（类别名称格式）")
                    else:
                        existing_classes = []
                        for class_id in self.class_ids:
                            if class_id in actual_classes:
                                existing_classes.append(class_id)
                            else:
                                class_name = self.class_id_to_name.get(class_id, class_id)
                                if class_name in actual_classes:
                                    existing_classes.append(class_name)
                                    self.class_id_to_name[class_name] = class_name
                        if len(existing_classes) > 0:
                            self.class_ids = existing_classes
                            if self.logger is not None:
                                self.logger.info(f"从类别列表中找到 {len(self.class_ids)} 个在数据目录中存在的类别")
                        else:
                            self.class_ids = sorted(actual_classes)
                            for class_name in self.class_ids:
                                self.class_id_to_name[class_name] = class_name
                            if self.logger is not None:
                                self.logger.warning(f"类别列表中的类别在数据目录中都不存在，使用数据目录中的实际类别: {len(self.class_ids)} 个")
        
        self.class_names = sorted(self.class_ids)
        self.class_to_idx = {class_id: idx for idx, class_id in enumerate(self.class_names)}
        
        if self.logger is not None:
            if class_list_file and os.path.exists(class_list_file):
                self.logger.info(f"从 {class_list_file} 加载了 {len(self.class_ids)} 个类别")
            else:
                self.logger.info(f"自动发现 {len(self.class_ids)} 个类别")
        else:
            print(f"加载了 {len(self.class_ids)} 个类别")
    
    def _get_cache_path(self):
        if self.class_list_file is None:
            class_list_str = "auto_discovered"
        else:
            class_list_str = os.path.basename(self.class_list_file)
        
        cache_key = hashlib.md5(
            f"{self.imagenet_root}_{self.split}_{class_list_str}".encode()
        ).hexdigest()[:8]
        project_root = '/data0/shiwei/shiwei/OSR'
        cache_dir = os.path.join(project_root, '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"imagenet_animal_fusion_{self.split}_{cache_key}.json")
    
    def _load_data(self):
        if self.use_cache:
            cache_path = self._get_cache_path()
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        if cached_data.get('imagenet_root') == self.imagenet_root:
                            cached_class_ids = set(cached_data.get('class_list_ids', []))
                            current_class_ids = set(self.class_ids)
                            
                            if current_class_ids.issubset(cached_class_ids) or cached_class_ids == current_class_ids:
                                cached_samples = cached_data.get('samples', [])
                                
                                if len(cached_samples) == 0:
                                    if self.logger is not None:
                                        self.logger.warning(f"缓存文件中的样本为空，重新扫描数据集...")
                                else:
                                    self.samples = cached_samples
                                    
                                    actual_class_ids = sorted(set([s['class_id'] for s in self.samples]))
                                    self.class_names = actual_class_ids
                                    self.class_to_idx = {class_id: idx for idx, class_id in enumerate(self.class_names)}
                                    
                                    for sample in self.samples:
                                        sample['class_idx'] = self.class_to_idx[sample['class_id']]
                                    
                                    if self.logger is not None:
                                        self.logger.info(f"从缓存加载 ImageNet {self.split} 数据集: {len(self.samples)} 个样本, {len(self.class_names)} 个类别")
                                    else:
                                        print(f"从缓存加载 ImageNet {self.split} 数据集: {len(self.samples)} 个样本, {len(self.class_names)} 个类别")
                                    return
                            else:
                                if self.logger is not None:
                                    self.logger.info(f"类别列表已更新（缓存: {len(cached_class_ids)} 个类别，当前: {len(current_class_ids)} 个类别），重新扫描数据集...")
                                else:
                                    print(f"类别列表已更新，重新扫描数据集...")
                except Exception as e:
                    if self.logger is not None:
                        self.logger.warning(f"加载缓存失败: {e}，重新扫描数据集...")
        
        split_dir = os.path.join(self.imagenet_root, self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"ImageNet {self.split} 目录不存在: {split_dir}")
        
        if self.logger is not None:
            self.logger.info(f"正在扫描 ImageNet {self.split} 数据集...")
        
        if self.split == 'unknow1':
            vlcs_domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007', 'NICO']
            
            if len(self.class_ids) == 0:
                all_classes = set()
                for domain in vlcs_domains:
                    domain_dir = os.path.join(split_dir, domain)
                    if os.path.exists(domain_dir):
                        for item in os.listdir(domain_dir):
                            item_path = os.path.join(domain_dir, item)
                            if os.path.isdir(item_path):
                                all_classes.add(item)
                self.class_ids = sorted(all_classes)
                for class_name in self.class_ids:
                    self.class_id_to_name[class_name] = class_name
                if self.logger is not None:
                    self.logger.info(f"从 unknow1 所有域中自动发现 {len(self.class_ids)} 个类别: {self.class_ids}")
            
            for domain in vlcs_domains:
                domain_dir = os.path.join(split_dir, domain)
                if not os.path.exists(domain_dir):
                    continue
                
                for class_id in self.class_ids:
                    class_dir = os.path.join(domain_dir, class_id)
                    if not os.path.exists(class_dir):
                        continue
                    
                    for root, dirs, files in os.walk(class_dir):
                        foreground_files = []
                        for f in files:
                            if f.lower().endswith(('_f.jpg', '_f.jpeg', '_f.png', '_f.JPEG', '_f.JPG', '_f.PNG')):
                                foreground_files.append(os.path.join(root, f))
                        
                        for fg_path in foreground_files:
                            bg_path = fg_path.replace('_f.', '_b.').replace('_F.', '_B.')
                            if not os.path.exists(bg_path):
                                base_name = os.path.splitext(fg_path)[0]
                                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                                    bg_path = base_name.replace('_f', '_b').replace('_F', '_B') + ext
                                    if os.path.exists(bg_path):
                                        break
                            
                            if os.path.exists(bg_path):
                                self.samples.append({
                                    'foreground_path': fg_path,
                                    'background_path': bg_path,
                                    'class_id': class_id,
                                    'class_name': self.class_id_to_name.get(class_id, class_id)
                                })
        else:
            for class_id in self.class_ids:
                class_dir = os.path.join(split_dir, class_id)
                if not os.path.exists(class_dir):
                    if self.logger is not None:
                        self.logger.warning(f"类别目录不存在: {class_dir}，跳过")
                    continue
                
                for root, dirs, files in os.walk(class_dir):
                    foreground_files = []
                    for f in files:
                        if f.lower().endswith(('_f.jpg', '_f.jpeg', '_f.png', '_f.JPEG', '_f.JPG', '_f.PNG')):
                            foreground_files.append(os.path.join(root, f))
                    
                    for fg_path in foreground_files:
                        bg_path = fg_path.replace('_f.', '_b.').replace('_F.', '_B.')
                        if not os.path.exists(bg_path):
                            base_name = os.path.splitext(fg_path)[0]
                            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                                bg_path = base_name.replace('_f', '_b').replace('_F', '_B') + ext
                                if os.path.exists(bg_path):
                                    break
                        
                        if os.path.exists(bg_path):
                            self.samples.append({
                                'foreground_path': fg_path,
                                'background_path': bg_path,
                                'class_id': class_id,
                                'class_name': self.class_id_to_name[class_id]
                            })
        
        actual_class_ids = sorted(set([s['class_id'] for s in self.samples]))
        self.class_names = actual_class_ids
        self.class_to_idx = {class_id: idx for idx, class_id in enumerate(self.class_names)}
        
        for sample in self.samples:
            sample['class_idx'] = self.class_to_idx[sample['class_id']]
        
        if self.use_cache:
            try:
                cached_data = {
                    'imagenet_root': self.imagenet_root,
                    'class_list_ids': self.class_ids,  # 保存类别列表，用于检测变化
                    'samples': self.samples
                }
                cache_path = self._get_cache_path()
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cached_data, f, indent=2, ensure_ascii=False)
                if self.logger is not None:
                    self.logger.info(f"已保存索引缓存到: {cache_path}")
            except Exception as e:
                if self.logger is not None:
                    self.logger.warning(f"保存缓存失败: {e}")
        
        missing_classes = set(self.class_ids) - set(actual_class_ids)
        if missing_classes and self.logger is not None:
            self.logger.warning(f"类别列表中有 {len(missing_classes)} 个类别在实际数据中不存在: {sorted(missing_classes)[:5]}{'...' if len(missing_classes) > 5 else ''}")
        
        if self.logger is not None:
            self.logger.info(f"ImageNet {self.split} 数据集加载完成: {len(self.samples)} 个样本, {len(self.class_names)} 个实际类别（类别列表中有 {len(self.class_ids)} 个类别）")
        else:
            print(f"ImageNet {self.split} 数据集加载完成: {len(self.samples)} 个样本, {len(self.class_names)} 个类别")
        
        if len(self.samples) == 0:
            raise ValueError(f"ImageNet {self.split} 数据集为空！请检查数据路径和类别列表。")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            foreground = Image.open(sample['foreground_path']).convert('RGB')
            background = Image.open(sample['background_path']).convert('RGB')
        except Exception as e:
            if self.logger is not None:
                self.logger.warning(f"无法加载图片 {sample['foreground_path']} 或 {sample['background_path']}: {e}")
            foreground = Image.new('RGB', (224, 224), color=(0, 0, 0))
            background = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.transform:
            foreground = self.transform(foreground)
            background = self.transform(background)
        
        return foreground, background, sample['class_idx']

